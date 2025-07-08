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

    def get_width(self) -> int:
        return self.width
    
    def get_height(self) -> int:
        return self.height

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

    @staticmethod
    def compute_texture_complexity(block: np.ndarray) -> Tuple[float, float]:
        """计算纹理复杂度：使用Sobel梯度"""
        if block.size == 0:
            return 0.0, 0.0
        
        block_f32 = block.astype(np.float32)
        grad_x = cv2.Sobel(block_f32, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(block_f32, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        texture_mean = float(np.mean(gradient_magnitude))
        texture_std = float(np.std(gradient_magnitude))
        return texture_mean, texture_std

    @staticmethod
    def compute_edge_density(block: np.ndarray) -> float:
        """计算边缘密度"""
        if block.size == 0:
            return 0.0
        
        edges = cv2.Canny(block, 50, 150)
        edge_density = float(np.sum(edges > 0) / edges.size)
        return edge_density

    @staticmethod
    def compute_residual_energy(block: np.ndarray) -> float:
        """计算残差能量（使用拉普拉斯算子近似高频内容）"""
        if block.size == 0:
            return 0.0
        
        block_f32 = block.astype(np.float32)
        laplacian = cv2.Laplacian(block_f32, cv2.CV_32F)
        residual_energy = float(np.var(laplacian))
        return residual_energy

    @staticmethod
    def compute_block_activity(block: np.ndarray) -> float:
        """计算块活动度（与均值的绝对差之和）"""
        if block.size == 0:
            return 0.0
        
        mean_val = np.mean(block)
        activity = float(np.mean(np.abs(block - mean_val)))
        return activity

    @staticmethod
    def compute_motion_features(mv_x: int, mv_y: int) -> Tuple[float, float]:
        """计算运动特征：幅度和角度"""
        mv_magnitude = float(np.sqrt(mv_x**2 + mv_y**2))
        mv_angle = float(np.arctan2(mv_y, mv_x)) if mv_magnitude > 0 else 0.0
        return mv_magnitude, mv_angle

    @staticmethod
    def compute_ssim(img1: np.ndarray, img2: np.ndarray, K1: float = 0.01, K2: float = 0.03, L: int = 255) -> float:
        """
        计算两个图像之间的SSIM (Structural Similarity Index)
        
        Args:
            img1, img2: 输入图像 (灰度图)
            K1, K2: SSIM算法常数
            L: 像素值动态范围 (通常为255)
        
        Returns:
            SSIM值 (0-1之间，1表示完全相同)
        """
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same dimensions")
        
        # 确保输入为float类型
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        # SSIM算法常数
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        
        # 计算均值
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # 计算方差和协方差
        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
        
        # 计算SSIM
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        
        ssim_map = numerator / denominator
        return float(np.mean(ssim_map))

    def ycrcb_ssim(
        self,
        frame_number: int,
        other_frame: np.ndarray,  # (3/2 * H, W)
    ) -> Tuple[float, float, float]:
        """
        计算YCbCr三个分量的SSIM
        
        Args:
            frame_number: 帧号
            other_frame: 编码后的帧 (yuv420p格式)
        
        Returns:
            (y_ssim, cb_ssim, cr_ssim): 三个分量的SSIM值
        """
        target_components = self.read_ycrcb_components(frame_number)
        if target_components is None:
            raise ValueError(f"Unable to read frame {frame_number} from the video.")

        if target_components.shape != other_frame.shape:
            raise ValueError("Dimension mismatch between video frame and reference frame components.")

        # 提取Y, Cb, Cr分量
        y_target = target_components[0:self.height, :]
        y_encoded = other_frame[0:self.height, :]
        
        cb_target = target_components[self.height:self.height + self.height // 4, :]
        cb_encoded = other_frame[self.height:self.height + self.height // 4, :]
        
        cr_target = target_components[self.height + self.height // 4:self.height + self.height // 2, :]
        cr_encoded = other_frame[self.height + self.height // 4:self.height + self.height // 2, :]

        # 计算各分量SSIM
        y_ssim = VideoReader.compute_ssim(y_target, y_encoded)
        cb_ssim = VideoReader.compute_ssim(cb_target, cb_encoded)
        cr_ssim = VideoReader.compute_ssim(cr_target, cr_encoded)

        return y_ssim, cb_ssim, cr_ssim


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