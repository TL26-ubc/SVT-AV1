import csv
import enum
from typing import Optional, Tuple

import cv2
import numpy as np

from typing import Optional, Tuple
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


class VideoComponent(enum.Enum):
    Y = "Y"
    Cb = "Cb"
    Cr = "Cr"


SB_SIZE = 64


class VideoReader:
    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {path}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.observation_max_values = {
            'y_variance': -float('inf'),
            'h_motion_vector': -float('inf'), 
            'v_motion_vector': -float('inf'),
            'gradient_magnitude': -float('inf')
        }

        self.visualization_data = {
            'y_variance': [],
            'h_motion_vector': [],
            'v_motion_vector': [],
            'gradient_magnitude': []
        }

    def collect_baseline_observation_stats(
        self, output_dir: str = "observation_analysis"
    ) -> dict:
       
        max_values = {
            'y_variance': -float('inf'),
            'h_motion_vector': -float('inf'), 
            'v_motion_vector': -float('inf'),
            'gradient_magnitude': -float('inf')
        }     

        os.makedirs(output_dir, exist_ok=True)

        self.visualization_data = {
            'y_variance': [],
            'h_motion_vector': [],
            'v_motion_vector': [],
            'gradient_magnitude': []
        }

        num_frames = self.get_frame_count()
        
        if num_frames == 0:
            raise ValueError("Video has no frames")
        
        valid_frames = 0
        frame_indices = []
        
        for frame_number in range(num_frames):
            try:
                frame_state = self.get_x_frame_state(frame_number)
                if frame_state == [[], [], [], []]:
                    continue
                    
                y_var_list, h_mv_list, v_mv_list, beta_list = frame_state

                self.visualization_data['y_variance'].append(y_var_list)
                self.visualization_data['h_motion_vector'].append(h_mv_list)
                self.visualization_data['v_motion_vector'].append(v_mv_list)
                self.visualization_data['gradient_magnitude'].append(beta_list)
                
                if y_var_list:
                    max_values['y_variance'] = max(
                        max_values['y_variance'], 
                        max(y_var_list)
                    )
               
                if h_mv_list:
                    max_abs_h = max(abs(v) for v in h_mv_list)
                    max_values['h_motion_vector'] = max(
                        max_values['h_motion_vector'], max_abs_h
                    )
                if v_mv_list:
                    max_abs_v = max(abs(v) for v in v_mv_list)
                    max_values['v_motion_vector'] = max(
                        max_values['v_motion_vector'],
                        max_abs_v
                    )
                if beta_list:
                    max_values['gradient_magnitude'] = max(
                        max_values['gradient_magnitude'], 
                        max(beta_list)
                    )
                frame_indices.append(frame_number)
                valid_frames += 1
                
            except Exception as e:
                print(f"Warning: Failed to process frame {frame_number}: {e}")
                continue
        
        if valid_frames == 0:
            raise ValueError("No valid frames found for collecting statistics")
        
        self.observation_max_values = max_values
        
        print(
            f"Baseline observation statistics collected from "
            f"{valid_frames}/{num_frames} valid frames:"
        )
        for key, value in max_values.items():
            print(f"  {key}: {value}")
        
        self._generate_visualization_plots(frame_indices, output_dir)
        return max_values

    def get_x_frame_state_normalized(self, frame_number) -> list[list[float]]:
       
        frame_state = self.get_x_frame_state(frame_number)
        if frame_state == [[], [], [], []]:
            return frame_state  
             
        y_var_list, h_mv_list, v_mv_list, beta_list = frame_state
        
        def normalize_variance_list(values, max_val):
            if max_val <= 0:
                return [0.0] * len(values)
            return [v / max_val for v in values]
        
        def normalize_motion_vector_list(values, max_abs_val):
            if max_abs_val <= 0:
                return [0.0] * len(values)
            return [v / max_abs_val for v in values]
        
        normalized_y_var = normalize_variance_list(
            y_var_list,
            self.observation_max_values['y_variance']
        )
        normalized_h_mv = normalize_motion_vector_list(
            h_mv_list,
            self.observation_max_values['h_motion_vector']
        ) 
        normalized_v_mv = normalize_motion_vector_list(
            v_mv_list,
            self.observation_max_values['v_motion_vector']
        )
        normalized_beta = normalize_variance_list(
            beta_list,
            self.observation_max_values['gradient_magnitude']
        )
        
        return [
            normalized_y_var,
            normalized_h_mv,
            normalized_v_mv,
            normalized_beta,
        ]
    
    def _generate_visualization_plots(self, frame_indices: list, output_dir: str):
        """生成观测值变化的可视化图表"""
        
        # 1. 时间序列图：显示每个特征的统计值变化
        self._plot_time_series_statistics(frame_indices, output_dir)
        
        # 2. 热力图：显示每个superblock的值变化
        self._plot_superblock_heatmaps(frame_indices, output_dir)
        
        # 3. 分布图：显示值的分布情况
        self._plot_value_distributions(output_dir)
        
        # 4. 相关性分析
        self._plot_feature_correlations(output_dir)
        
        print(f"Visualization plots saved to {output_dir}/")

    def _plot_time_series_statistics(self, frame_indices: list, output_dir: str):
        """绘制时间序列统计图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Observation Features Statistics Over Time', fontsize=16)
        
        feature_names = [
            'y_variance',
            'h_motion_vector',
            'v_motion_vector',
            'gradient_magnitude'
        ]
        feature_titles = [
            'Y-Component Variance',
            'Horizontal Motion Vector',
            'Vertical Motion Vector',
            'Gradient Magnitude'
        ]
        
        for idx, (feature, title) in enumerate(
            zip(feature_names, feature_titles)
        ):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            data = self.visualization_data[feature]
            if not data:
                continue
                
            # 计算每帧的统计值
            means = [np.mean(frame_data) for frame_data in data]
            maxs = [np.max(frame_data) for frame_data in data]
            mins = [np.min(frame_data) for frame_data in data]
            stds = [np.std(frame_data) for frame_data in data]
            
            # 绘制统计线
            ax.plot(frame_indices, means, label='Mean', linewidth=2, alpha=0.8)
            ax.plot(frame_indices, maxs, label='Max', linewidth=1, alpha=0.7)
            ax.plot(frame_indices, mins, label='Min', linewidth=1, alpha=0.7)
            ax.fill_between(frame_indices, 
                           [m - s for m, s in zip(means, stds)], 
                           [m + s for m, s in zip(means, stds)], 
                           alpha=0.3, label='±1 Std')
            
            ax.set_title(title)
            ax.set_xlabel('Frame Number')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/time_series_statistics.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_superblock_heatmaps(self, frame_indices: list, output_dir: str):
        """绘制superblock热力图"""
        
        num_frames = len(frame_indices)
        num_superblocks = len(self.visualization_data['y_variance'][0]) if self.visualization_data['y_variance'] else 0
        
        if num_superblocks == 0:
            return
        
        # 计算superblock的空间排列
        num_blocks_h = (self.height + SB_SIZE - 1) // SB_SIZE
        num_blocks_w = (self.width + SB_SIZE - 1) // SB_SIZE
        
        feature_names = ['y_variance', 'h_motion_vector', 'v_motion_vector', 'gradient_magnitude']
        feature_titles = ['Y-Component Variance', 'Horizontal Motion Vector', 'Vertical Motion Vector', 'Gradient Magnitude']
        
        for feature, title in zip(feature_names, feature_titles):
            if not self.visualization_data[feature]:
                continue
                
            # 创建热力图数据：每个superblock在所有帧上的平均值
            heatmap_data = np.zeros((num_blocks_h, num_blocks_w))
            
            # 计算每个superblock的平均值
            sb_averages = []
            for sb_idx in range(num_superblocks):
                values = [frame_data[sb_idx] for frame_data in self.visualization_data[feature] if sb_idx < len(frame_data)]
                if values:
                    sb_averages.append(np.mean(values))
                else:
                    sb_averages.append(0)
            
            # 将1D数据重新排列为2D网格
            for i, avg_val in enumerate(sb_averages):
                row = i // num_blocks_w
                col = i % num_blocks_w
                if row < num_blocks_h:
                    heatmap_data[row, col] = avg_val
            
            # 绘制热力图
            plt.figure(figsize=(12, 8))
            
            # 使用不同的颜色映射
            if 'motion' in feature:
                cmap = 'RdBu_r'  # 红蓝色映射，适合正负值
                center = 0
            else:
                cmap = 'viridis'  # 普通色映射
                center = None
            
            sns.heatmap(heatmap_data, 
                       annot=False, 
                       cmap=cmap, 
                       center=center,
                       cbar_kws={'label': 'Average Value'})
            
            plt.title(f'{title} - Average Values Across All Frames\n(Superblock Spatial Layout)')
            plt.xlabel('Superblock Column')
            plt.ylabel('Superblock Row')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/heatmap_{feature}.png", dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_value_distributions(self, output_dir: str):
        """绘制值分布图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Distribution of Observation Features', fontsize=16)
        
        feature_names = ['y_variance', 'h_motion_vector', 'v_motion_vector', 'gradient_magnitude']
        feature_titles = ['Y-Component Variance', 'Horizontal Motion Vector', 'Vertical Motion Vector', 'Gradient Magnitude']
        
        for idx, (feature, title) in enumerate(zip(feature_names, feature_titles)):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            data = self.visualization_data[feature]
            if not data:
                continue
            
            # 将所有值展平
            all_values = [val for frame_data in data for val in frame_data]
            
            if all_values:
                # 绘制直方图和密度图
                ax.hist(all_values, bins=50, alpha=0.7, density=True, edgecolor='black')
                
                # 添加统计信息
                mean_val = np.mean(all_values)
                std_val = np.std(all_values)
                median_val = np.median(all_values)
                
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
                ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
                
                ax.set_title(f'{title}\n(μ={mean_val:.3f}, σ={std_val:.3f})')
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/value_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_feature_correlations(self, output_dir: str):
        """绘制特征相关性分析"""
        # 准备数据
        correlation_data = {}
        feature_names = ['y_variance', 'h_motion_vector', 'v_motion_vector', 'gradient_magnitude']
        
        # 计算每帧每个特征的平均值
        for feature in feature_names:
            data = self.visualization_data[feature]
            if data:
                correlation_data[feature] = [np.mean(frame_data) for frame_data in data]
        
        if len(correlation_data) < 2:
            return
        
        # 创建DataFrame用于相关性分析
        import pandas as pd
        df = pd.DataFrame(correlation_data)
        
        # 计算相关性矩阵
        correlation_matrix = df.corr()
        
        # 绘制相关性热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.3f',
                   cbar_kws={'label': 'Correlation Coefficient'})
        
        plt.title('Feature Correlation Matrix\n(Frame-wise Average Values)')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_correlations.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制散点图矩阵
        if len(correlation_data) >= 2:
            fig, axes = plt.subplots(len(feature_names), len(feature_names), figsize=(16, 16))
            fig.suptitle('Feature Scatter Plot Matrix', fontsize=16)
            
            for i, feature1 in enumerate(feature_names):
                for j, feature2 in enumerate(feature_names):
                    ax = axes[i, j]
                    
                    if i == j:
                        # 对角线：显示分布
                        if feature1 in correlation_data:
                            ax.hist(correlation_data[feature1], bins=20, alpha=0.7)
                            ax.set_title(feature1.replace('_', ' ').title())
                    else:
                        # 非对角线：显示散点图
                        if feature1 in correlation_data and feature2 in correlation_data:
                            ax.scatter(correlation_data[feature2], correlation_data[feature1], alpha=0.6, s=10)
                            
                            # 计算并显示相关系数
                            corr = correlation_matrix.loc[feature1, feature2]
                            ax.text(0.05, 0.95, f'r={corr:.3f}', transform=ax.transAxes, 
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    if i == len(feature_names) - 1:
                        ax.set_xlabel(feature2.replace('_', ' ').title())
                    if j == 0:
                        ax.set_ylabel(feature1.replace('_', ' ').title())
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/scatter_plot_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()

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

    def get_x_frame_state(self, frame_number) -> list[list[float]]:
        """
        Extracts the state of a video frame based on superblock information.

        Args:
            frame (np.ndarray): The video frame.
            block_size (int): Size of the blocks to be processed. Should be 64 in SVT-AV1.

        Returns:
            a list of lists containing superblock information:
                0 Y-component variance of all superblocks in the frame
                1 Horizontal and
                2 vertical difference of all superblocks in the frame
                3 Gradient magnitude of all superblocks in the frame
        """
        frame = self.read_frame(frame_number)
        if frame is None:
            # no further processing is needed, just return empty lists
            return [[], [], [], []]

        h, w = frame.shape[:2]
        y_comp_list = []
        h_mv_list = []
        v_mv_list = []
        beta_list = []

        for y in range(0, h, SB_SIZE):
            for x in range(0, w, SB_SIZE):  # follow encoder order, x changes first
                y_end = min(y + SB_SIZE, h)
                x_end = min(x + SB_SIZE, w)
                sb = frame[y:y_end, x:x_end]
                if sb.size == 0:
                    continue

                sb_y_var = np.var(sb[:, :, 0])  # Y-component variance
                sb_x_mv = np.mean(sb[:, :, 1])  # Horizontal motion vector
                sb_y_mv = np.mean(sb[:, :, 2])  # Vertical motion vector
                beta = np.mean(np.abs(sb))  # Example metric

                y_comp_list.append(sb_y_var)
                h_mv_list.append(sb_x_mv)
                v_mv_list.append(sb_y_mv)
                beta_list.append(beta)

        return [y_comp_list, h_mv_list, v_mv_list, beta_list]

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
