import csv
import enum
from typing import Literal, Optional, Tuple, TypeAlias, cast

import cv2
import numpy as np
from numpy import ndarray

import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from pyencoder.environment.constants import SB_SIZE

class VideoComponent(enum.Enum):
    Y = "Y"
    Cb = "Cb"
    Cr = "Cr"

class VideoReader:
    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {path}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read_frame(self, frame_number) -> Optional[np.ndarray]: # (H, W, 3)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()

    def get_resolution(self) -> Tuple[int, int]:
        return self.width, self.height

    def read_ycrcb_components(self, frame_number: int) -> Optional[np.ndarray]: # (H, W, 3)
        rgb_frame = self.read_frame(frame_number=frame_number)
        if rgb_frame is None:
            return None
        ycrcb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2YCrCb)
        return ycrcb_frame

    def get_frame_count(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def render_frame_number(self, frame_number: int):
        frame = self.read_frame(frame_number=frame_number)
        if frame is not None:
            self.render_frame(frame)

    def render_frame(self, frame: np.ndarray):
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # sb info
    def get_num_superblock(self):
        num_blocks_h = (self.height + SB_SIZE - 1) // SB_SIZE
        num_blocks_w = (self.width + SB_SIZE - 1) // SB_SIZE
        return num_blocks_h * num_blocks_w

    def ycrcb_psnr(
        self,
        frame_number: int,
        other_frame: np.ndarray, # (H, W, 3)
        baseline_heighest_psnr
    ):
        """
        frame number
        other frame: (y,cb,cr)
        should be same size
        """
        target_components = self.read_ycrcb_components(frame_number)
        if target_components is None:
            raise ValueError(f"Unable to read frame {frame_number} from the video.")

        if target_components.shape != other_frame.shape:
            raise ValueError("Dimension mismatch between video frame and reference frame components.")

        y_psnr = VideoReader.compute_psnr(target_components[:, :, 0], other_frame[:, :, 0], baseline_heighest_psnr["y"])
        cb_psnr = VideoReader.compute_psnr(target_components[:, :, 1], other_frame[:, :, 1], baseline_heighest_psnr['cb'])
        cr_psnr = VideoReader.compute_psnr(target_components[:, :, 2], other_frame[:, :, 2], baseline_heighest_psnr['cr'])

        # render the image for debug 
        # target_bgr = cv2.cvtColor(target_components, cv2.COLOR_YCrCb2BGR)
        # other_bgr = cv2.cvtColor(other_frame, cv2.COLOR_YCrCb2BGR)
        # cv2.imwrite(f"target_{frame_number}.png", target_bgr)
        # cv2.imwrite(f"other_frame_{frame_number}.png", other_bgr)
        return y_psnr, cb_psnr, cr_psnr

    @staticmethod
    def render_single_component(
        component_array: np.ndarray, component_type: VideoComponent
    ):
        cv2.imshow(str(component_type.value), component_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def render_components(y: np.ndarray, cb: np.ndarray, cr: np.ndarray):
        # OpenCV uses Y, Cr, Cb order
        ycrcb_image = cv2.merge((y, cr, cb))

        bgr_image = cv2.cvtColor(ycrcb_image, cv2.COLOR_YCrCb2BGR)
        cv2.imshow("BGR", bgr_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def compute_psnr(target: np.ndarray, reference: np.ndarray, baseline_heighest_psnr: float = 100.0):
        psnr = cv2.PSNR(target, reference)
        return psnr if np.isfinite(psnr) else baseline_heighest_psnr


# # simple test
# if __name__ == "__main__":
#     reader = VideoReader("/home/tom/tmp/playground/akiyo_qcif.y4m")

#     reader.get_resolution()
#     reader.get_frame_count()
#     y, cb, cr = reader.read_ycrcb_components(1)

#     # Flatten arrays and write to CSV
#     with open("frame1_ycrcb.csv", "w", newline="") as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(["Component", "Row", "Col", "Value"])
#         for comp_name, comp_array in zip(["Y", "Cb", "Cr"], [y, cb, cr]):
#             for row in range(comp_array.shape[0]):
#                 for col in range(comp_array.shape[1]):
#                     writer.writerow([comp_name, row, col, int(comp_array[row, col])])

#     # reader.render_single_component(y, VideoComponent.Y)

#     VideoReader.render_components(y, cb, cr)
