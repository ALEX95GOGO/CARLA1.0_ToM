import os
from util import read_rgb_image_strict, overlay_attention, get_avaliable_gpu
from inference import infer_sequence

from ai_sa_model import SequenceEncoder_human_sa

def main():
    # get available device
    device = get_avaliable_gpu()

    # path to model weights
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "model_ckpt", "human_sa_guided_risk_predictor.pth")

    # instantiate model and load weights
    model = SequenceEncoder_human_sa()
    model.load_model(model_path, device=device)

    # read 3 test frames
    #frames = [read_rgb_image_strict(f"{current_dir}/test_frame/frame{i}.png") for i in range(3)]
    frames = [read_rgb_image_strict(f"{current_dir}/test_frame_custom/{i:06d}.png") for i in range(550, 556, 2)]

    #import pdb; pdb.set_trace()
    # run inference
    risk, attn_maps = infer_sequence(model, frames, device=device)

    # print risk value and save attention maps
    print("Risk:", risk)
    overlay_attention(frames[0], attn_maps[0], alpha=0.5, save_path=f"{current_dir}/out/attention_map_c2.png")

if __name__ == "__main__":
    main()
