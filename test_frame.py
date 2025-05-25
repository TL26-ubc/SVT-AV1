import enum
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import csv


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

    def read_ycbcr_components(self, frame_number: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        frame = self.read()
        if frame is None:
            return None
        ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycbcr)
        return y, cb, cr  # Return in standard order

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
    def render_single_component(component_array: np.ndarray, component_type: VideoComponent):
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


# simple test
if __name__ == "__main__":
    reader = VideoReader("/home/tom/tmp/playground/akiyo_qcif.y4m")

    reader.get_resolution()
    reader.get_frame_count()
    y, cb, cr = reader.read_ycbcr_components(1)


    # Flatten arrays and write to CSV
    with open("frame1_ycbcr.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Component", "Row", "Col", "Value"])
        for comp_name, comp_array in zip(["Y", "Cb", "Cr"], [y, cb, cr]):
            for row in range(comp_array.shape[0]):
                for col in range(comp_array.shape[1]):
                    writer.writerow([comp_name, row, col, int(comp_array[row, col])])

    # reader.render_single_component(y, VideoComponent.Y)

    VideoReader.render_components(y, cb, cr)
