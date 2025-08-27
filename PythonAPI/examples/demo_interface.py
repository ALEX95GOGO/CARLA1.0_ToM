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
import cv2
import numpy as np

from collections import deque

queue_size = 3
frame_queue = deque(maxlen=queue_size)

device = get_avaliable_gpu()

# path to model weights
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "ai_sa", "model_ckpt", "pedestrian_predict.pth")
#model_path = os.path.join(current_dir, "ai_sa", "model_ckpt", "human_sa_guided_risk_predictor.pth")
# instantiate model and load weights
model = SequenceEncoder_human_sa()
model.load_model(model_path, device=device)

# >>> NEW: globals for UI thread-safe display
latest_frame = None
latest_overlay = None
latest_risk = None
show_overlay = True


def camera_callback(image):
    global latest_rgb
    # Convert CARLA raw image to numpy array
    #image.convert(carla.ColorConverter.CityScapesPalette)

    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA format
    # Convert BGRA to BGR for OpenCV
    frame = array[:, :, :3]
    latest_rgb = frame

def semantic_camera_callback(image):
    global latest_frame, latest_overlay, latest_risk  # >>> NEW
    # Convert CARLA raw image to numpy array
    image.convert(carla.ColorConverter.CityScapesPalette)

    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA format
    # Convert BGRA to BGR for OpenCV
    frame = array[:, :, :3]
    #print(frame.max())
    #[:,:,::-1].copy()
    frame_queue.append(frame[:,:,::-1].copy())

    # >>> CHANGED: don't call imshow here; just compute & store results
    latest_frame = frame
    latest_overlay = None
    latest_risk = None

    if len(frame_queue) == 3:
        risk, attn_maps = infer_sequence(model, frame_queue, device=device)
        latest_risk = float(risk)
        # print risk value
        print("Risk:", latest_risk)
        # build overlay for UI thread to draw
        latest_overlay = overlay_attention(latest_rgb, 1-attn_maps[0], alpha=0.5)


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


def main():
    global show_overlay  # >>> NEW

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--host", metavar="H", default="127.0.0.1",
                           help="IP of the host server (default: 127.0.0.1)")
    argparser.add_argument("-p", "--port", metavar="P", default=2000, type=int,
                           help="TCP port to listen to (default: 2000)")
    argparser.add_argument("-rh", "--roshost", metavar="Rh", default="192.168.86.33",
                           help="IP of the host ROS server (default: 192.168.86.33)")
    argparser.add_argument("-rp", "--rosport", metavar="Rp", default=11311,
                           help="TCP port for ROS server (default: 11311)")
    argparser.add_argument("-sh", "--selfhost", metavar="Sh", default="192.168.86.123",
                           help="IP of the ROS node (this machine) (default: 192.168.86.123)")
    args = argparser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    sync_mode = False  # synchronous mode
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
    # Set camera attributes (adjust as needed)
    semantic_camera_bp.set_attribute('image_size_x', '900')
    semantic_camera_bp.set_attribute('image_size_y', '256')
    semantic_camera_bp.set_attribute('fov', '105')

    # Define relative transform (position & rotation) to ego vehicle
    camera_transform = carla.Transform(carla.Location(x=1.5, z=1))  # in front and above ego

    # Spawn the camera sensor attached to ego_vehicle
    semantic_camera_sensor = world.spawn_actor(semantic_camera_bp, camera_transform, attach_to=ego_vehicle)

    camera_bp = blueprint_library.find('sensor.camera.rgb')

    # Set camera attributes (adjust as needed)
    camera_bp.set_attribute('image_size_x', '900')
    camera_bp.set_attribute('image_size_y', '256')
    camera_bp.set_attribute('fov', '105')

    # Define relative transform (position & rotation) to ego vehicle
    camera_transform = carla.Transform(carla.Location(x=1.5, z=1))  # in front and above ego

    # Spawn the camera sensor attached to ego_vehicle
    camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)

    def publish_and_print(data):
        sensor.update(data)
        if rospy is not None:
            msg: String = create_ros_msg(sensor)
            pub.publish(msg)
        print(sensor.data['focus_actor_name'])

    # camera + ML callback
    camera_sensor.listen(camera_callback)
    semantic_camera_sensor.listen(semantic_camera_callback)
    sensor.ego_sensor.listen(publish_and_print)

    # >>> NEW: create window once; draw from main thread
    cv2.namedWindow('Ego Camera View', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Ego Camera View', 900, 256)

    try:
        while True:
            if sync_mode:
                world.tick()
            else:
                world.wait_for_tick()

            # --- UI drawing (non-blocking) ---
            disp = None
            if show_overlay and latest_overlay is not None:
                disp = latest_overlay.copy()
            #elif latest_frame is not None:
            #   disp = latest_frame.copy()


            if disp is not None:
                if latest_risk is not None:
                    cv2.putText(disp, f"Predicted Risk: {latest_risk:.3f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
                if sensor.data is not None:
                    cv2.putText(disp, f"Human Focus info: {sensor.data['focus_actor_name']}", (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('Ego Camera View', disp)

            # keys: 'o' toggle overlay, 'q' quit
            k = cv2.waitKey(10) & 0xFF
            if k == ord('o'):
                show_overlay = not show_overlay
            elif k == ord('q'):
                break

    finally:
        # >>> NEW: cleanup OpenCV and CARLA actors
        try:
            camera_sensor.stop()
            camera_sensor.destroy()
        except Exception:
            pass
        if sync_mode:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        cv2.destroyAllWindows()  # >>> NEW


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print("\ndone.")
