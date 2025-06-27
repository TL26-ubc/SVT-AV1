import csv
import enum
from typing import Optional, Tuple

import cv2
import numpy as np
from numpy import ndarray

from typing import Optional, Tuple
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pyencoder.states.__templete import State_templete


class VideoComponent(enum.Enum):
    Y = "Y"
    Cb = "Cb"
    Cr = "Cr"


SB_SIZE = 64


class VideoReader:
    def __init__(self, path: str, state_wrapper: State_templete = None):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {path}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        assert state_wrapper is not None, "State wrapper must be provided."
        self.state_wrapper = state_wrapper
        state_wrapper.initialize(
            video_reader=self,
            SB_SIZE=SB_SIZE
        )
        
    def get_x_frame_state_normalized(self, frame_number) -> ndarray:
       
        """
        Get the state of the x-th frame, normalized.
        """
        frame = self.read_frame(frame_number=frame_number)
        return self.state_wrapper.get_observation(frame, SB_SIZE=SB_SIZE)

    def read(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        return frame if ret else None

    def read_frame(self, frame_number):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        return self.read()

    def release(self):
        self.cap.release()

    def get_resolution(self) -> Tuple[int, int]:
        return self.width, self.height

    def read_ycrcb_components(
        self, frame_number: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        frame = self.read_frame(frame_number=frame_number)
        if frame is None:
            return None
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        return ycrcb  # Return in standard order

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

    # sb infor
    def get_num_superblock(self):
        num_blocks_h = (self.height + SB_SIZE - 1) // SB_SIZE
        num_blocks_w = (self.width + SB_SIZE - 1) // SB_SIZE
        return num_blocks_h * num_blocks_w

    def ycrcb_psnr(
        self,
        frame_number: int,
        other_frame: tuple[np.ndarray, np.ndarray, np.ndarray],
        baseline_heighest_psnr
    ):
        """
        frame number
        other frame: (y,cb,cr)
        should be same size
        """
        target_components = self.read_ycrcb_components(frame_number)
        if target_components is None:
            raise ValueError(
                f"Unable to read frame {frame_number} from the video."
            )

        if target_components.shape != other_frame.shape:
            raise ValueError(
                "Dimension mismatch between video frame and "
                "reference frame components."
            )

        # VideoReader.render_single_component(other_frame[0], VideoComponent.Y)
        y_psnr = VideoReader.compute_psnr(target_components[0], other_frame[0], baseline_heighest_psnr["y"])
        cb_psnr = VideoReader.compute_psnr(target_components[1], other_frame[1], baseline_heighest_psnr['cb'])
        cr_psnr = VideoReader.compute_psnr(target_components[2], other_frame[2], baseline_heighest_psnr['cr'])

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
    def compute_psnr(target, reference, baseline_heighest_psnr: float = 100.0):
        mse = np.mean((target.astype(np.float32) - reference.astype(np.float32)) ** 2)
        if mse == 0:
            # cannot return inf, as it will cause issues in rl training
            return baseline_heighest_psnr
        return 10 * np.log10((255.0**2) / mse)


# simple test
if __name__ == "__main__":
    reader = VideoReader("/home/tom/tmp/playground/akiyo_qcif.y4m")

    reader.get_resolution()
    reader.get_frame_count()
    y, cb, cr = reader.read_ycrcb_components(1)

    # Flatten arrays and write to CSV
    with open("frame1_ycrcb.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Component", "Row", "Col", "Value"])
        for comp_name, comp_array in zip(["Y", "Cb", "Cr"], [y, cb, cr]):
            for row in range(comp_array.shape[0]):
                for col in range(comp_array.shape[1]):
                    writer.writerow([comp_name, row, col, int(comp_array[row, col])])

    # reader.render_single_component(y, VideoComponent.Y)

    VideoReader.render_components(y, cb, cr)
