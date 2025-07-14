from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class YUVFrame:
    y_plane: np.ndarray # (w, h)
    u_plane: np.ndarray # (w // 2, h // 2)
    v_plane: np.ndarray # (w // 2, h // 2)

class VideoUtils:
    @staticmethod
    def get_yuv_planes(frame: np.ndarray, width: int, height: int) -> YUVFrame:
        """
        Split a flattened *YUV420p* frame into Y, U, V planes.

        Frame layout (planar, 8 bit):
            Y  :  w × h           bytes
            U  : (w/2) × (h/2)    bytes
            V  : (w/2) × (h/2)    bytes
        """
        y_size  = width * height
        uv_size = (width // 2) * (height // 2)
        
        expected = y_size + 2 * uv_size
        if frame.size != y_size + 2 * uv_size:
            raise ValueError(
                f"Unexpected frame length. Got {frame.size} bytes, "
                f"expected {expected} for {width}×{height} YUV420p."
            )

        linear_frame = frame.ravel()
        y_plane = linear_frame[:y_size].reshape((height, width))
        u_plane = linear_frame[y_size : y_size + uv_size].reshape((height // 2, width // 2))
        v_plane = linear_frame[y_size + uv_size :].reshape((height // 2, width // 2))

        return YUVFrame(y_plane, u_plane, v_plane)
    
    @staticmethod
    def compute_psnr(target: np.ndarray, reference: np.ndarray) -> float:
        return cv2.PSNR(target, reference)

    @staticmethod
    def compute_mse(target: np.ndarray, reference: np.ndarray) -> float:
        return np.mean((target.astype(np.float32) - reference.astype(np.float32)) ** 2).astype(float)

    @staticmethod
    def compute_rmse(target: np.ndarray, reference: np.ndarray) -> float:
        return np.sqrt(VideoUtils.compute_mse(target, reference))
    
    @staticmethod
    def compute_texture_complexity(block: np.ndarray) -> tuple[float, float]:
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
        if block.size == 0:
            return 0.0
        
        edges = cv2.Canny(block, 50, 150)
        edge_density = float(np.sum(edges > 0) / edges.size)
        return edge_density

    @staticmethod
    def compute_residual_energy(block: np.ndarray) -> float:
        if block.size == 0:
            return 0.0
        
        block_f32 = block.astype(np.float32)
        laplacian = cv2.Laplacian(block_f32, cv2.CV_32F)
        residual_energy = float(np.var(laplacian))
        return residual_energy

    @staticmethod
    def compute_block_activity(block: np.ndarray) -> float:
        if block.size == 0:
            return 0.0
        
        mean_val = np.mean(block)
        activity = float(np.mean(np.abs(block - mean_val)))
        return activity

    @staticmethod
    def compute_motion_features(mv_x: int, mv_y: int) -> tuple[float, float]:
        mv_magnitude = float(np.sqrt(mv_x**2 + mv_y**2))
        mv_angle = float(np.arctan2(mv_y, mv_x)) if mv_magnitude > 0 else 0.0
        return mv_magnitude, mv_angle

    @staticmethod
    def compute_ssim(img1: np.ndarray, img2: np.ndarray, K1: float = 0.01, K2: float = 0.03, L: int = 255) -> float:
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same dimensions")
        
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
        
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        
        ssim_map = numerator / denominator
        return float(np.mean(ssim_map))
    
