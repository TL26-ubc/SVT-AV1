import enum
import av
import cv2
import numpy as np
from av1gym.environment.constants import SB_SIZE

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

    def _ensure_frame_count(self) -> int:
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
        return self._frame_count

    def _get_frame_at_index(self, frame_number) -> av.VideoFrame | None:
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

    def read_frame_raw(self, frame_number) -> np.ndarray: 
        # The current frame in yuv420p format, shape (3/2 * H, W).
        av_frame = self._get_frame_at_index(frame_number)
        if av_frame is None:
            raise RuntimeError('This frame should exist')
        
        # yuv420p format conversion
        return av_frame.to_ndarray(format='yuv420p')

    def release(self):
        self.container.close()
        self._frames_cache.clear()

    def get_resolution(self) -> tuple[int, int]:
        return self.width, self.height

    def read_frame(self, frame_number: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]: # (w, h), (w // 2, h // 2), (w // 2, h // 2)
        ycrcb_frame = self.read_frame_raw(frame_number)
        return self.get_yuv_planes(ycrcb_frame)

    def get_frame_count(self) -> int:
        return self._ensure_frame_count()

    # sb info
    def get_superblock_dims(self) -> tuple[int, int]:
        num_blocks_h = (self.height + SB_SIZE - 1) // SB_SIZE
        num_blocks_w = (self.width + SB_SIZE - 1) // SB_SIZE
        return num_blocks_w, num_blocks_h

    @staticmethod
    def compute_psnr(target: np.ndarray, reference: np.ndarray) -> float:
        return cv2.PSNR(target, reference)

    @staticmethod
    def compute_mse(target: np.ndarray, reference: np.ndarray) -> float:
        return np.mean((target.astype(np.float32) - reference.astype(np.float32)) ** 2).astype(float)
    
    def get_yuv_planes(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        w, h = self.width, self.height
        y_plane = frame[0 : w*h].reshape((h, w))
        u_plane = frame[w*h : w*h + (w//2)*(h//2)].reshape((h//2, w//2))
        v_plane = frame[w*h + (w//2)*(h//2) : ].reshape((h//2, w//2))
        return y_plane, u_plane, v_plane
