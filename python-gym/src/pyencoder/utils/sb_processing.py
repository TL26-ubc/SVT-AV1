from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class SuperBlockInfo:
    sb_org_x: int
    sb_org_y: int
    sb_width: int
    sb_height: int
    sb_qindex: int
    sb_x_mv: int
    sb_y_mv: int
    beta: float

def psnr_block(block1, block2):
    mse = np.mean((block1.astype(np.float32) - block2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255 * 255 / mse)

def per_superblock_psnr(
    frame1, frame2, block_size=64
    ) -> list[list[float]]:
    h, w = frame1.shape[:2]
    psnr_map = []
    for y in range(0, h, block_size):
        row = []
        for x in range(0, w, block_size):
            y_end = min(y + block_size, h)
            x_end = min(x + block_size, w)
            block1 = frame1[y:y_end, x:x_end]
            block2 = frame2[y:y_end, x:x_end]
            if block1.shape == block2.shape and block1.size > 0:
                row.append(psnr_block(block1, block2))
            else:
                row.append(None)  # or handle edge blocks as needed
        psnr_map.append(row)
    return psnr_map

def get_psnr_list(
    original_video: cv2.VideoCapture,
    modified_video: cv2.VideoCapture
) -> list[float]:
    psnr_list = []
    while True:
        ret1, frame1 = original_video.read()
        ret2, frame2 = modified_video.read()
        if not ret1 or not ret2:
            break
        psnr = cv2.PSNR(frame1, frame2)
        psnr_list.append(psnr)
    return psnr_list
    
def get_diff(
    original_video: cv2.VideoCapture,
    modified_video: cv2.VideoCapture
)-> list[float]:
    
    # returns the PSNR for each frame in the video for now
    frame_psnr = get_psnr_list(original_video, modified_video)
    
    return frame_psnr