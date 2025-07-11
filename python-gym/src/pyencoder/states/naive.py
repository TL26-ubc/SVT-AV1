from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
from pyencoder import SuperBlockInfo
from pyencoder.environment.av1_runner import Observation
from pyencoder.utils.video_reader import VideoReader

from .abstract import AbstractState


class NaiveState(AbstractState):
    def __init__(
        self,
        video_reader: VideoReader,
        baseline_observations: list[Observation],
        sb_size: int = 64,
        include_neighborhood: bool = True,
        **kwargs: Any
    ):
        self.sb_size = sb_size
        self.num_sb = video_reader.get_num_superblock()
        self.frame_count = video_reader.get_frame_count()
        self.video_reader = video_reader
        self.include_neighborhood = include_neighborhood

        # Video properties for progress tracking
        self.video_width = video_reader.get_width()
        self.video_height = video_reader.get_height()
        self.sb_cols = (self.video_width + sb_size - 1) // sb_size
        self.sb_rows = (self.video_height + sb_size - 1) // sb_size

        # Historical data storage
        self.frame_history = []  # Store past frame quality metrics
        self.bits_history = []  # Store cumulative bits usage
        self.current_gop_start = 0  # Track GOP boundaries

        array_length = self.get_observation_length()
        self.max_values = np.full(array_length, -np.inf, dtype=np.float32)
        for raw_obs in baseline_observations:
            frame = video_reader.read_frame(frame_number=raw_obs.picture_number)
            if frame is None or len(frame) == 0:
                continue
            obs = self.get_observation(
                frame, raw_obs.superblocks, raw_obs.frame_type, raw_obs.picture_number
            )
            self.max_values = np.maximum(self.max_values, obs)

    def get_observation(
        self,
        frame: np.ndarray,  # The current frame in yuv420p format, shape (3/2 * H, W).
        sbs: list[SuperBlockInfo],
        frame_type: int,
        picture_number: int,
        **kwargs
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        assert h % 3 == 0, "Height must be a multiple of 3 for yuv420p format"
        h, w = h // 3 * 2, w  # Adjust for yuv420p format
        y_comp_list = []
        h_mv_list = []
        v_mv_list = []
        qindex_list = []

        # 初始化临时特征列表
        self.temp_texture_mean_list = []
        self.temp_texture_std_list = []
        self.temp_edge_density_list = []
        self.temp_residual_energy_list = []
        self.temp_block_activity_list = []
        self.temp_mv_magnitude_list = []
        self.temp_mv_angle_list = []
        self.temp_cb_var_list = []
        self.temp_cr_var_list = []

        sb_idx = 0
        for y in range(0, h, self.sb_size):
            for x in range(0, w, self.sb_size):  # follow encoder order, x changes first
                y_end = min(y + self.sb_size, h)
                x_end = min(x + self.sb_size, w)
                sb_y_component = frame[y:y_end, x:x_end]
                # sb_cb_component = frame[h + y // 2:h + y_end // 2, x:x_end]
                # sb_cr_component = frame[h + h // 2 + y // 2:h + h // 2 + y_end // 2, x:x_end]
                if sb_y_component.size == 0:
                    continue

                # Extract Cb and Cr components
                sb_cb_component = frame[h + y // 2 : h + y_end // 2, x:x_end]
                sb_cr_component = frame[
                    h + h // 2 + y // 2 : h + h // 2 + y_end // 2, x:x_end
                ]

                # Original features (保留原有特征)
                sb_y_var = np.var(sb_y_component)
                sb_x_mv = sbs[sb_idx]["sb_x_mv"]
                sb_y_mv = sbs[sb_idx]["sb_y_mv"]
                sb_qindex = sbs[sb_idx]["sb_qindex"]

                # 1. 内容复杂度特征 (Content Complexity Features)
                # 纹理复杂度
                texture_mean, texture_std = (
                    self.video_reader.compute_texture_complexity(sb_y_component)
                )
                # 边缘密度
                edge_density = self.video_reader.compute_edge_density(sb_y_component)
                # 残差能量
                residual_energy = self.video_reader.compute_residual_energy(
                    sb_y_component
                )
                # 块活动度
                block_activity = self.video_reader.compute_block_activity(
                    sb_y_component
                )
                # 运动矢量特征（幅度和角度）
                mv_magnitude, mv_angle = self.video_reader.compute_motion_features(
                    sb_x_mv, sb_y_mv
                )
                # Cb, Cr 方差
                sb_cb_var = np.var(sb_cb_component) if sb_cb_component.size > 0 else 0.0
                sb_cr_var = np.var(sb_cr_component) if sb_cr_component.size > 0 else 0.0

                y_comp_list.append(sb_y_var)
                h_mv_list.append(sb_x_mv)
                v_mv_list.append(sb_y_mv)
                qindex_list.append(sb_qindex)

                # 添加新的复杂度特征到临时列表
                self.temp_texture_mean_list.append(texture_mean)
                self.temp_texture_std_list.append(texture_std)
                self.temp_edge_density_list.append(edge_density)
                self.temp_residual_energy_list.append(residual_energy)
                self.temp_block_activity_list.append(block_activity)
                self.temp_mv_magnitude_list.append(mv_magnitude)
                self.temp_mv_angle_list.append(mv_angle)
                self.temp_cb_var_list.append(sb_cb_var)
                self.temp_cr_var_list.append(sb_cr_var)
                sb_idx += 1

        # 构建增强的observation（保留原有特征+新增特征）
        # 原有特征（4个）
        obs_lists = [y_comp_list, h_mv_list, v_mv_list, qindex_list]

        # 新增内容复杂度特征（9个）
        obs_lists.extend(
            [
                self.temp_texture_mean_list,
                self.temp_texture_std_list,
                self.temp_edge_density_list,
                self.temp_residual_energy_list,
                self.temp_block_activity_list,
                self.temp_mv_magnitude_list,
                self.temp_mv_angle_list,
                self.temp_cb_var_list,
                self.temp_cr_var_list,
            ]
        )

        # 2. 编码进展和历史信息特征
        progress_features = self._get_progress_features(picture_number)

        # 3. 帧/块类型信息
        type_features = self._get_type_features(frame_type)

        # 4. 质量反馈特征
        quality_features = self._get_quality_features()

        # 5. 邻域信息（如果启用）
        neighborhood_features = []
        if self.include_neighborhood:
            neighborhood_features = self._get_neighborhood_features(sbs)

        # 构建最终observation
        obs = np.array(obs_lists, dtype=np.float32).flatten()

        # 添加全局特征
        global_features = (
            [
                float(picture_number),  # 原有的picture number
            ]
            + progress_features
            + type_features
            + quality_features
            + neighborhood_features
        )

        obs = np.append(obs, global_features)

        # check for inf or nan values and handle them
        obs = np.where(np.isfinite(obs), obs, self.max_values[: len(obs)])
        return obs

    def _get_progress_features(self, picture_number: int) -> List[float]:
        """2. 编码进展和历史信息特征"""
        features = []

        # 当前进度比例
        progress_ratio = float(picture_number) / max(1, self.frame_count - 1)
        features.append(progress_ratio)

        # 累计比特数
        total_bits = sum(self.bits_history)
        features.append(float(total_bits))

        # 平均每帧比特数
        avg_bits = total_bits / max(1, len(self.bits_history))
        features.append(float(avg_bits))

        # GOP位置（假设GOP大小为16）
        gop_position = picture_number % 16
        features.append(float(gop_position))

        # 剩余帧数比例
        remaining_ratio = max(0, self.frame_count - picture_number - 1) / max(
            1, self.frame_count
        )
        features.append(remaining_ratio)

        return features

    def _get_type_features(self, frame_type: int) -> List[float]:
        """3. 帧/块类型信息"""
        features = []

        # 帧类型 one-hot 编码（支持I/P/B等类型）
        frame_types = [0.0, 0.0, 0.0, 0.0]  # 最多支持4种帧类型
        if 0 <= frame_type < len(frame_types):
            frame_types[frame_type] = 1.0
        features.extend(frame_types)

        return features

    def _get_quality_features(self) -> List[float]:
        """4. 质量反馈特征（包含PSNR和SSIM）"""
        features = []

        # 前一帧的PSNR值
        if len(self.frame_history) > 0:
            prev_frame = self.frame_history[-1]
            features.extend(
                [
                    float(prev_frame.get("y_psnr", 0.0)),
                    float(prev_frame.get("cb_psnr", 0.0)),
                    float(prev_frame.get("cr_psnr", 0.0)),
                ]
            )
            # 前一帧的SSIM值
            features.extend(
                [
                    float(prev_frame.get("y_ssim", 0.0)),
                    float(prev_frame.get("cb_ssim", 0.0)),
                    float(prev_frame.get("cr_ssim", 0.0)),
                ]
            )
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # 最近几帧的平均PSNR
        if len(self.frame_history) >= 3:
            recent_frames = self.frame_history[-3:]
            avg_y_psnr = np.mean([f.get("y_psnr", 0.0) for f in recent_frames])
            avg_cb_psnr = np.mean([f.get("cb_psnr", 0.0) for f in recent_frames])
            avg_cr_psnr = np.mean([f.get("cr_psnr", 0.0) for f in recent_frames])
            # 最近几帧的平均SSIM
            avg_y_ssim = np.mean([f.get("y_ssim", 0.0) for f in recent_frames])
            avg_cb_ssim = np.mean([f.get("cb_ssim", 0.0) for f in recent_frames])
            avg_cr_ssim = np.mean([f.get("cr_ssim", 0.0) for f in recent_frames])
            features.extend(
                [
                    float(avg_y_psnr),
                    float(avg_cb_psnr),
                    float(avg_cr_psnr),
                    float(avg_y_ssim),
                    float(avg_cb_ssim),
                    float(avg_cr_ssim),
                ]
            )
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        return features

    def _get_neighborhood_features(self, sbs: List[SuperBlockInfo]) -> List[float]:
        """5. 邻域信息特征（空间一致性）"""
        if not self.include_neighborhood:
            return []

        features = []

        # 计算所有superblock的平均QP和运动矢量
        if len(sbs) > 0:
            avg_qp = np.mean([sb["sb_qindex"] for sb in sbs])
            avg_mv_x = np.mean([sb["sb_x_mv"] for sb in sbs])
            avg_mv_y = np.mean([sb["sb_y_mv"] for sb in sbs])

            # QP和运动矢量的标准差（表示空间一致性）
            std_qp = np.std([sb["sb_qindex"] for sb in sbs])
            std_mv_x = np.std([sb["sb_x_mv"] for sb in sbs])
            std_mv_y = np.std([sb["sb_y_mv"] for sb in sbs])

            features.extend(
                [
                    float(avg_qp),
                    float(avg_mv_x),
                    float(avg_mv_y),
                    float(std_qp),
                    float(std_mv_x),
                    float(std_mv_y),
                ]
            )
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        return features

    def update_history(
        self, picture_number: int, quality_metrics: Dict, bitstream_size: int
    ):
        """更新历史信息，在环境中每帧结束后调用"""
        self.frame_history.append(
            {
                "picture_number": picture_number,
                "y_psnr": quality_metrics.get("y_psnr", 0.0),
                "cb_psnr": quality_metrics.get("cb_psnr", 0.0),
                "cr_psnr": quality_metrics.get("cr_psnr", 0.0),
                "y_ssim": quality_metrics.get("y_ssim", 0.0),
                "cb_ssim": quality_metrics.get("cb_ssim", 0.0),
                "cr_ssim": quality_metrics.get("cr_ssim", 0.0),
            }
        )

        self.bits_history.append(bitstream_size)

        # 保持历史记录大小合理
        max_history = 50
        if len(self.frame_history) > max_history:
            self.frame_history = self.frame_history[-max_history:]
        if len(self.bits_history) > max_history:
            self.bits_history = self.bits_history[-max_history:]

    def get_observation_length(self) -> int:
        """
        计算观测向量的总长度

        原有特征: 4 * num_sb (Y方差, X运动矢量, Y运动矢量, QP索引)
        新增内容复杂度特征: 9 * num_sb (纹理均值/标准差, 边缘密度, 残差能量, 块活动度, 运动幅度/角度, Cb/Cr方差)
        全局特征: 1 (图像编号) + 5 (进展特征) + 4 (帧类型) + 12 (质量反馈:PSNR+SSIM) + 6 (邻域特征，可选)
        """
        # 每个superblock的特征数：原有4个 + 新增9个 = 13个
        sb_features = 13 * self.num_sb

        # 全局特征数量
        global_features = 1  # picture_number
        global_features += 5  # progress features
        global_features += 4  # type features
        global_features += 12  # quality features (6 PSNR + 6 SSIM)

        if self.include_neighborhood:
            global_features += 6  # neighborhood features

        return sb_features + global_features

    def get_observation_space(self) -> gym.spaces.Space:
        """
        Get the observation space of the state.

        Returns:
            gym.spaces.Space: A gymnasium Space object representing the observation space.
        """
        return gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.video_height, self.video_width, 3),  # Assuming RGB observation
            dtype=np.uint8,
        )
