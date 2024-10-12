import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import numpy as np
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    image = cv2.imread("/home/yujiannan/图片/2024-10-12_17-18.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    color_mask = np.zeros_like(image)
    predictor.set_image(image)
    masks, _, _ = predictor.predict()
    # mask叠加到原图显示
    mask = masks.transpose(1, 2, 0)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    color_mask[mask > 0] = (0, 255,255)
    image = cv2.addWeighted(image, 0.5, color_mask, 0.5, gamma=0)
    cv2.imshow("image", image)
    cv2.waitKey(0)
#

# import torch
# from sam2.build_sam import build_sam2_video_predictor
#
# checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
# predictor = build_sam2_video_predictor(model_cfg, checkpoint)
#
# with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
#     state = predictor.init_state(<your_video>)
#
#     # add new prompts and instantly get the output on the same frame
#     frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, <your_prompts>):
#
#     # propagate the prompts to get masklets throughout the video
#     for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
#         ...