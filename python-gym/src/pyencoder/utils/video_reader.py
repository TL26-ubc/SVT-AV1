import enum
from typing import Literal, Optional, Tuple, TypeAlias, cast
import av
import cv2
import numpy as np
from numpy import ndarray
from pyencoder.environment.constants import SB_SIZE

class VideoComponent(enum.Enum):
    Y = "Y"
    Cb = "Cb"
    Cr = "Cr"

class VideoReader:
    def __init__(self, path: str):
        self.path = path
        self.container = av.open(path)
        self.video_stream = self.container.streams.video[0]
        if self.video_stream is None:
            raise ValueError(f"Cannot open video file: {path}")
        self.width = self.video_stream.width
        self.height = self.video_stream.height
        self._frame_count = None
        self._frames_cache = {}  # Cache frames for random access

    def _ensure_frame_count(self):
        """Ensure frame count is calculated"""
        if self._frame_count is None:
            if self.video_stream.frames > 0:
                self._frame_count = self.video_stream.frames
            else:
                # Manual count if metadata unreliable
                count = 0
                for _ in self.container.decode(video=0):
                    count += 1
                self._frame_count = count
                self.container.seek(0)

    def _get_frame_at_index(self, frame_number) -> Optional[av.VideoFrame]:
        """Get PyAV frame at specific index"""
        if frame_number in self._frames_cache:
            return self._frames_cache[frame_number]
        
        # Reset container and iterate to target frame
        self.container.seek(0)
        current_index = 0
        for frame in self.container.decode(video=0):
            if current_index == frame_number:
                self._frames_cache[frame_number] = frame
                return frame
            current_index += 1
        return None

    def read_frame(self, frame_number) -> Optional[np.ndarray]: 
        # The current frame in yuv420p format, shape (3/2 * H, W).
        av_frame = self._get_frame_at_index(frame_number)
        if av_frame is None:
            return None
        
        # yuv420p format conversion
        return av_frame.to_ndarray(format='yuv420p')

    def release(self):
        self.container.close()
        self._frames_cache.clear()

    def get_resolution(self) -> Tuple[int, int]:
        return self.width, self.height

    def read_ycrcb_components(self, frame_number: int) -> Optional[np.ndarray]: # (3/2 * H, W) in yuv420p format
        av_frame = self._get_frame_at_index(frame_number)
        if av_frame is None:
            return None
        
        # Direct YUV420p conversion like in the reference code
        ycrcb_frame = av_frame.to_ndarray(format='yuv420p')
        return ycrcb_frame

    def get_frame_count(self) -> int:
        self._ensure_frame_count()
        return self._frame_count

    # sb info
    def get_num_superblock(self):
        num_blocks_h = (self.height + SB_SIZE - 1) // SB_SIZE
        num_blocks_w = (self.width + SB_SIZE - 1) // SB_SIZE
        return num_blocks_h * num_blocks_w

    def ycrcb_psnr(
        self,
        frame_number: int,
        other_frame: np.ndarray, # (3/2 * H, W)
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

        y_psnr = VideoReader.compute_psnr(target_components[0:self.height, :], other_frame[0:self.height, :], 
                                           baseline_heighest_psnr["y"])
        cb_psnr = VideoReader.compute_psnr(target_components[self.height:self.height + self.height // 4, :], 
                                           other_frame[self.height:self.height + self.height // 4, :],
                                           baseline_heighest_psnr['cb'])
        cr_psnr = VideoReader.compute_psnr(target_components[self.height + self.height // 4:self.height + self.height // 2, :],
                                           other_frame[self.height + self.height // 4:self.height + self.height // 2, :],
                                           baseline_heighest_psnr['cr'])

        return y_psnr, cb_psnr, cr_psnr

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