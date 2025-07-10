from dataclasses import dataclass
import av
import cv2
import numpy as np
from av1gym.environment.constants import SB_SIZE

@dataclass
class YUVFrame:
    y_plane: np.ndarray # (w, h)
    u_plane: np.ndarray # (w // 2, h // 2)
    v_plane: np.ndarray # (w // 2, h // 2)

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

    def read_frame(self, frame_number: int) -> YUVFrame:
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
    
    def get_yuv_planes(self, frame: np.ndarray) -> YUVFrame:
        """
        Split a flattened *YUV420p* frame into Y, U, V planes.

        Frame layout (planar, 8 bit):
            Y  :  w × h           bytes
            U  : (w/2) × (h/2)    bytes
            V  : (w/2) × (h/2)    bytes
        """
        y_size  = self.width * self.height
        uv_size = (self.width // 2) * (self.height // 2)
        expected = y_size + 2 * uv_size

        if frame.size != expected:
            raise ValueError(
                f"Unexpected frame length. Got {frame.size} bytes, "
                f"expected {expected} for {self.width}×{self.height} YUV420p."
            )

        linear_frame = frame.ravel()
        y_plane = linear_frame[:y_size].reshape((self.height, self.width))
        u_plane = linear_frame[y_size : y_size + uv_size].reshape((self.height // 2, self.width // 2))
        v_plane = linear_frame[y_size + uv_size :].reshape((self.height // 2, self.width // 2))

        return YUVFrame(y_plane, u_plane, v_plane)
