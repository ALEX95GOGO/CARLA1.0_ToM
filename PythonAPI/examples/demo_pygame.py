import argparse
import numpy as np
import time
import sys
import os
from DReyeVR_utils import DReyeVRSensor
from DReyeVR_utils import find_ego_vehicle
import os
sys.path.append('ai_sa')
from util import read_rgb_image_strict, overlay_attention, get_avaliable_gpu
from inference import infer_sequence

from ai_sa_model import SequenceEncoder_human_sa

try:
    import rospy
    from std_msgs.msg import String
except ImportError:
    rospy = None
    String = None
    print("Rospy not initialized. Unable to use ROS for logging")

import carla
import cv2  # keep if overlay_attention uses it internally
import numpy as np
import pygame  # >>> NEW

from collections import deque
import torch
from torchvision import transforms

import whisper
import pyaudio
import numpy as np
import queue
import threading
import time
from typing import Callable, Optional
import re
import asyncio

CHILD = [sys.executable, "-u", "sound_recording.py"]  # -u => unbuffered

TRANSCRIPTION_RE = re.compile(r"\s*Transcription:\s*(.*)")
latest_transcript = ""

def handle_stdout_line(line: str):
    # Route transcription vs. logs
    m = TRANSCRIPTION_RE.search(line)
    global latest_transcript
    if m:
        text = m.group(1)
        latest_transcript = text          # <-- store for HUD

        print(f"[PARENT] New transcription: {text}")
        # TODO: send to GUI / websocket / queue / etc.
    else:
        print(f"[child stdout] {line}")

def handle_stderr_line(line: str):
    print(f"[child stderr] {line}")

async def stream_reader(stream: asyncio.StreamReader, handler):
    while True:
        line = await stream.readline()
        if not line:
            break
        # decode & strip newlines
        handler(line.decode(errors="replace").rstrip("\r\n"))
        
queue_size = 3
frame_queue = deque(maxlen=queue_size)

device = get_avaliable_gpu()

# path to model weights
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "ai_sa", "model_ckpt", "pedestrian_predict.pth")

model = SequenceEncoder_human_sa()
model.load_model(model_path, device=device)

# globals for UI thread-safe display
latest_frame = None
latest_overlay = None
latest_risk = None
show_overlay = True
latest_rgb = None  # >>> NEW

def camera_callback(image):
    global latest_rgb
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA
    frame = array[:, :, :3]  # BGR
    latest_rgb = frame

'''
def semantic_camera_callback(image):
    global latest_frame, latest_overlay, latest_risk
    image.convert(carla.ColorConverter.CityScapesPalette)

    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA
    frame = array[:, :, :3]  # BGR
    #[:, :, ::-1]
    frame = torch.from_numpy(frame[:, :, ::-1].copy()).permute(2,0,1).float()/255.0
    transform = transforms.Compose([transforms.Resize((224,224))])
    frame_queue.append(transform(frame))  # RGB copy for your model if needed

    latest_frame = frame
    #latest_overlay = None
    #latest_risk = None

    if len(frame_queue) == 3 and latest_rgb is not None:
        risk, attn_maps = infer_sequence(model, list(frame_queue), device=device)
        latest_risk = float(risk)
        #print("Risk:", latest_risk)
        # overlay_attention likely returns BGR; keep as-is for now
        latest_overlay = overlay_attention(latest_rgb, 1 - attn_maps[2], alpha=0.3)
'''
transform_resize_224 = transforms.Resize((224, 224))

def semantic_camera_callback(image):
    global latest_frame, latest_overlay, latest_risk
    image.convert(carla.ColorConverter.CityScapesPalette)

    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    frame = array[:, :, :3]  # BGR

    # build a CPU tensor without a grad graph
    frame_t = torch.from_numpy(frame[:, :, ::-1].copy()).permute(2,0,1).float() / 255.0

    # CHANGED: reuse the global transform (no Compose per frame)
    frame_t = transform_resize_224(frame_t)

    frame_queue.append(frame_t)  # CPU tensor only
    latest_frame = frame

    # --- inference block ---
    if len(frame_queue) == 3 and latest_rgb is not None:
        # NEW: disable autograd so no computation graph is kept
        with torch.inference_mode():
            risk, attn_maps = infer_sequence(model, list(frame_queue), device=device)

        # ensure only small CPU data goes to globals
        latest_risk = float(risk) if not isinstance(risk, float) else risk

        att = attn_maps[2]
        if isinstance(att, torch.Tensor):
            att = att.detach().cpu().numpy()   # CHANGED: move off GPU and drop graph

        latest_overlay = overlay_attention(latest_rgb, 1.0 - att, alpha=0.3)

        # OPTIONAL tiny GC hint for long runs on CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    

def create_ros_msg(ego_sensor: DReyeVRSensor, delim: str = "; "):
    assert rospy is not None and String is not None
    s = "rosT=" + str(rospy.get_time()) + delim
    for key in ego_sensor.data:
        s += f"{key}={ego_sensor.data[key]}{delim}"
    return String(s)

def init_ros_pub(IP_SELF, IP_ROSMASTER, PORT_ROSMASTER):
    assert rospy is not None and String is not None
    os.environ["ROS_IP"] = IP_SELF
    os.environ["ROS_MASTER_URI"] = "http://" + IP_ROSMASTER + ":" + str(PORT_ROSMASTER)
    try:
        rospy.set_param("bebop_ip", IP_ROSMASTER)
        rospy.init_node("dreyevr_node")
        return rospy.Publisher("dreyevr_pub", String, queue_size=10)
    except ConnectionRefusedError:
        print("RospyError: Could not initialize rospy connection")
        sys.exit(1)

def numpy_bgr_to_pygame_surface(bgr_frame):
    """Convert BGR numpy image (H,W,3) -> Pygame Surface (RGB)."""
    rgb = bgr_frame[:, :, ::-1]  # BGR -> RGB
    h, w, _ = rgb.shape
    return pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")

async def main():
    # Platform-specific process group so we can terminate cleanly
    kwargs = {}
    if os.name == "nt":
        # Windows: create a new process group (CTRL_BREAK_EVENT will target it)
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        kwargs["creationflags"] = CREATE_NEW_PROCESS_GROUP
    else:
        # POSIX: put child in its own process group
        kwargs["preexec_fn"] = os.setsid

    proc = await asyncio.create_subprocess_exec(
        *CHILD,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **kwargs
    )

    print(f"[PARENT] spawned pid={proc.pid}")

    # Start readers
    t_out = asyncio.create_task(stream_reader(proc.stdout, handle_stdout_line))
    t_err = asyncio.create_task(stream_reader(proc.stderr, handle_stderr_line))


    global show_overlay

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--host", metavar="H", default="127.0.0.1")
    argparser.add_argument("-p", "--port", metavar="P", default=2000, type=int)
    argparser.add_argument("-rh", "--roshost", metavar="Rh", default="192.168.86.33")
    argparser.add_argument("-rp", "--rosport", metavar="Rp", default=11311)
    argparser.add_argument("-sh", "--selfhost", metavar="Sh", default="192.168.86.123")
    args = argparser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(100.0)
    sync_mode = True
    np.random.seed(int(time.time()))

    if rospy is not None:
        IP_SELF = args.selfhost
        IP_ROSMASTER = args.roshost
        PORT_ROSMASTER = args.rosport
        pub = init_ros_pub(IP_SELF, IP_ROSMASTER, PORT_ROSMASTER)

    world = client.get_world()
    sensor = DReyeVRSensor(world)
    ego_vehicle = find_ego_vehicle(world)
    blueprint_library = world.get_blueprint_library()

    semantic_camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    semantic_camera_bp.set_attribute('image_size_x', '1600')
    semantic_camera_bp.set_attribute('image_size_y', '512')
    semantic_camera_bp.set_attribute('fov', '105')

    camera_transform = carla.Transform(carla.Location(x=1.5, z=1))
    semantic_camera_sensor = world.spawn_actor(semantic_camera_bp, camera_transform, attach_to=ego_vehicle)

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1600')
    camera_bp.set_attribute('image_size_y', '512')
    camera_bp.set_attribute('fov', '105')
    camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)


    def publish_and_print(data):
        sensor.update(data)
        if rospy is not None:
            msg: String = create_ros_msg(sensor)
            pub.publish(msg)
        #print(sensor.data['focus_actor_name'])

    camera_sensor.listen(camera_callback)
    semantic_camera_sensor.listen(semantic_camera_callback)
    sensor.ego_sensor.listen(publish_and_print)

    # ============== Pygame display (replaces cv2 window) ==============
    pygame.init()  # >>> NEW
    screen_w, screen_h = 1200, 400
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Ego Camera View (Pygame)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)  # for on-screen text

    try:
        running = True
        while running:
            if sync_mode:
                world.tick()
            else:
                world.wait_for_tick()

            # Handle events / keys
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_o:
                        show_overlay = not show_overlay
                    elif event.key == pygame.K_q:
                        running = False

            screen.fill((0, 0, 0))

            # Choose image to display
            disp = None
            if show_overlay and latest_overlay is not None:
                disp = latest_overlay
            #elif latest_frame is not None:
            #    disp = latest_frame

            if disp is not None:
                surf = numpy_bgr_to_pygame_surface(disp)  # BGR->RGB for Pygame
                # scale to window if needed
                surf = pygame.transform.smoothscale(surf, (screen_w, screen_h))
                screen.blit(surf, (0, 0))

                # Draw text overlays
                if latest_risk is not None:
                    txt = font.render(f"Predicted Risk: {latest_risk:.3f}", True, (255, 255, 0))
                    screen.blit(txt, (10, 10))
                if sensor.data is not None and 'focus_actor_name' in sensor.data:
                    actor_name = sensor.data['focus_actor_name']
                    txt_actor = None
                    if 'Road' in actor_name:
                        txt_actor = 'Road'
                    if 'Vehicle' in actor_name:
                        txt_actor = 'Vehicle'
                    if 'Walker' in actor_name:
                        txt_actor = 'Pedestrian'
                    if 'Light' in actor_name:
                        txt_actor = 'TrafficLight'
                    if 'SM' in actor_name:
                        txt_actor = 'Building'
                    
                    txt2 = font.render(f"Human Focus info: {txt_actor}", True, (255, 255, 0))
                    screen.blit(txt2, (10, 40))
                    #print(latest_transcript)
                    txt3 = font.render(f"Human Speech: {latest_transcript}", True, (255, 255, 0))
                    screen.blit(txt3, (10, 80))

            pygame.display.flip()
            clock.tick(60)  # ~60 FPS
            # --- yield to asyncio once per frame so tasks run smoothly ---
            await asyncio.sleep(0)                # add at end of while running loop
    finally:
        try:
            camera_sensor.stop()
            camera_sensor.destroy()
        except Exception:
            pass
        try:
            semantic_camera_sensor.stop()
            semantic_camera_sensor.destroy()
        except Exception:
            pass
        if sync_mode:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        pygame.quit()  # >>> NEW
        try:
            transcriber.stop()
        except Exception:
            pass

async def terminate(proc: asyncio.subprocess.Process):
    if proc.returncode is not None:
        return
    try:
        if os.name == "nt":
            # Send CTRL_BREAK to the process group
            os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
        else:
            # Send SIGTERM to the whole group
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
            return
        except asyncio.TimeoutError:
            pass
        # Force kill
        if os.name == "nt":
            proc.kill()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        await proc.wait()
    except ProcessLookupError:
        pass

if __name__ == "__main__":
    asyncio.run(main())
