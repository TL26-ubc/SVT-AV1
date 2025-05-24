import enum
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


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

    def read(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()

    def get_resolution(self) -> Tuple[int, int]:
        return self.width, self.height

    def read_ycbcr_components(
        self, frame_number: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        frame = self.read()
        if frame is None:
            return None
        ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycbcr)
        return y, cb, cr  # Return in standard order

    def read_ycbcr_components_chopped(
        self,
        frame_number: int,
        left_top: Tuple[int, int],
        right_bottom: Tuple[int, int],
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        frame = self.read()
        if frame is None:
            return None
        ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycbcr)
        y = y[left_top[1] : right_bottom[1], left_top[0] : right_bottom[0]]
        cb = cb[left_top[1] : right_bottom[1], left_top[0] : right_bottom[0]]
        cr = cr[left_top[1] : right_bottom[1], left_top[0] : right_bottom[0]]
        return y, cb, cr

    def get_frame_count(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def render_frame_number(self, frame_number: int):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        frame = self.read()
        if frame is not None:
            self.render_frame(frame)

    def render_frame(self, frame: np.ndarray):
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
        ycbcr_image = cv2.merge((y, cr, cb))

        bgr_image = cv2.cvtColor(ycbcr_image, cv2.COLOR_YCrCb2BGR)
        cv2.imshow("BGR", bgr_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
        if original.shape != compressed.shape:
            raise ValueError("Original and compressed images must have the same shape.")

        mse = np.mean(
            (original.astype(np.float64) - compressed.astype(np.float64)) ** 2
        )
        if mse == 0:
            return float("inf")
        PIXEL_MAX = 255.0
        return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

    @staticmethod
    def calculate_ssim(original: np.ndarray, compressed: np.ndarray) -> float:
        if original.shape != compressed.shape:
            raise ValueError("Original and compressed images must have the same shape.")
        return ssim(
            original,
            compressed,
            data_range=original.max() - original.min(),
            multichannel=True,
        )

    @staticmethod
    def calculate_mse(original: np.ndarray, compressed: np.ndarray) -> float:
        if original.shape != compressed.shape:
            raise ValueError("Original and compressed images must have the same shape.")
        return np.mean((original - compressed) ** 2)


# simple test
if __name__ == "__main__":
    reader = VideoReader("Data\\akiyo_qcif.y4m")

    reader.get_resolution()
    reader.get_frame_count()
    y, cb, cr = reader.read_ycbcr_components(1)

    reader.render_single_component(y, VideoComponent.Y)

    VideoReader.render_components(y, cb, cr)
