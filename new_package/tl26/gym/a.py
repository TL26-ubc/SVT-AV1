import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
import time
from typing import Dict, Any, Tuple, List, Optional

class VideoEncoderEnv(gym.Env):
    """
    A Gymnasium environment for video encoder optimization.
    
    This environment serves as an interface between RL algorithms and the SVT-AV1 encoder.
    It manages the state representation, action selection, and reward calculation for optimizing
    video encoding parameters using reinforcement learning.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 config: Dict[str, Any] = None,
                 render_mode: Optional[str] = None):
        """
        Initialize the video encoder environment.
        
        Args:
            config: Configuration dictionary for the environment:
                - frame_width: Width of the video frames
                - frame_height: Height of the video frames
                - qp_range: Range of quantization parameters [min, max]
                - target_bitrate: Target bitrate in bits per second
                - reward_weights: Weights for different components of the reward function
                  (e.g., {'psnr': 1.0, 'bitrate': -0.5, 'ssim': 0.3})
            render_mode: Rendering mode ('human' or None)
        """
        super().__init__()
        
        # Default configuration
        self.config = {
            'frame_width': 1920,
            'frame_height': 1080,
            'qp_range': [-63, 63],
            'qp_offset_range': [-8, 8],
            'target_bitrate': 5000000,  # 5 Mbps
            'reward_weights': {
                'psnr': 1.0,      # Higher PSNR is better
                'bitrate': -0.5,  # Lower bitrate is better (negative weight)
                'ssim': 0.3       # Higher SSIM is better
            },
            'features': [
                'sb_index',
                'sb_origin_x',
                'sb_origin_y',
                'sb_qp',
                'sb_final_blk_cnt',
                'mi_row_start',
                'mi_row_end',
                'mi_col_start',
                'mi_col_end',
                'tg_horz_boundary',
                'tile_row',
                'tile_col',
                'tile_rs_index',
                'encoder_bit_depth',
                'qindex',
                'beta',
                'slice_type'
            ]
        }
        
        # Update config with provided values
        if config:
            self.config.update(config)
            
        self.render_mode = render_mode
        
        # Initialize state
        self.frame_idx = 0
        self.total_frames = 0
        self.current_frame_state = {}
        self.current_block_idx = 0
        self.blocks_in_frame = []
        self.encoding_stats = {}
        
        # Action space: QP offset for each superblock
        # Typically QP offsets are in a small range like -8 to +8
        qp_min_offset, qp_max_offset = self.config['qp_offset_range']
        n_actions = qp_max_offset - qp_min_offset + 1
        self.action_space = spaces.Discrete(n_actions)
        self.qp_min_offset = qp_min_offset
        
        # Observation space: State features for superblocks
        # Features include position, surrounding blocks, texture complexity, etc.
        num_features = len(self.config['features'])
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(num_features,), 
            dtype=np.float32
        )
        
        # Statistics
        self.episode_rewards = []
        self.episode_psnrs = []
        self.episode_bitrates = []
        self.episode_ssims = []
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Tracking the current episode
        self.current_episode = 0
        self.steps_in_episode = 0
        
        self.logger.info("VideoEncoderEnv initialized")

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment at the start of a new episode.
        
        Args:
            seed: Random seed
            options: Additional options for reset
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        self.frame_idx = 0
        self.current_block_idx = 0
        self.blocks_in_frame = []
        self.encoding_stats = {}
        self.steps_in_episode = 0
        
        # Reset episode statistics
        self.episode_rewards = []
        self.episode_psnrs = []
        self.episode_bitrates = []
        self.episode_ssims = []
        
        self.current_episode += 1
        self.logger.info(f"Episode {self.current_episode} started")
        
        # The initial observation would typically be the first frame's first block
        # Since we're simulating an interface to SVT-AV1, we'll return a placeholder
        # In practice, this would be filled with actual block features from the encoder
        observation = np.zeros(len(self.config['features']), dtype=np.float32)
        
        info = {
            "episode": self.current_episode,
            "frame_idx": self.frame_idx,
            "block_idx": self.current_block_idx
        }
        
        return observation, info

    def step(self, action):
        """
        Take a step in the environment by applying the selected QP offset to the current block.
        
        Args:
            action: QP offset index (mapped to actual offset value)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Map action index to actual QP offset
        qp_offset = action + self.qp_min_offset
        
        # In a real implementation, this would apply the QP offset to the encoder
        # and get back the next state (next block or next frame)
        
        # Simulating encoder feedback
        # In practice, this would come from the actual encoder
        reward = self._calculate_reward(qp_offset)
        self.steps_in_episode += 1
        
        # Update block and frame indices
        self.current_block_idx += 1
        if self.current_block_idx >= len(self.blocks_in_frame):
            # Move to next frame
            self.current_block_idx = 0
            self.frame_idx += 1
            # Get new blocks for the next frame (in practice, from the encoder)
            self.blocks_in_frame = self._get_next_frame_blocks()
        
        # Check if episode is done
        terminated = self.frame_idx >= self.total_frames
        truncated = False
        
        # Get next observation (next block features)
        observation = self._get_next_observation()
        
        # Prepare info dict
        info = {
            "episode": self.current_episode,
            "frame_idx": self.frame_idx,
            "block_idx": self.current_block_idx,
            "qp_offset": qp_offset,
            "reward": reward
        }
        
        # Optional rendering
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info

    def _calculate_reward(self, qp_offset):
        """
        Calculate the reward based on encoding quality and bitrate.
        
        In practice, this would use actual metrics from the encoder after
        encoding with the selected QP offset.
        
        Args:
            qp_offset: Applied QP offset
            
        Returns:
            reward: Calculated reward value
        """
        # Placeholder for demonstration
        # In a real implementation, these would come from actual encoding results
        
        # Simulate PSNR (higher is better)
        # QP offsets affect quality: negative offsets (lower QP) give higher quality
        base_psnr = 35.0  # Baseline PSNR for the current content
        psnr_factor = -0.5  # Each point of QP change affects PSNR
        psnr = base_psnr + psnr_factor * qp_offset
        
        # Simulate bitrate (bits per pixel)
        # Lower QP (negative offset) increases bitrate
        base_bitrate = 0.1  # Baseline bits per pixel
        bitrate_factor = 0.02  # Each point of QP change affects bitrate
        bitrate = base_bitrate * (1.0 - bitrate_factor * qp_offset)
        
        # Simulate SSIM (higher is better)
        base_ssim = 0.95
        ssim_factor = -0.01
        ssim = base_ssim + ssim_factor * qp_offset
        
        # Calculate reward using weighted components
        weights = self.config['reward_weights']
        reward = (
            weights.get('psnr', 1.0) * psnr +
            weights.get('bitrate', -0.5) * bitrate +
            weights.get('ssim', 0.3) * ssim
        )
        
        # Record statistics
        self.episode_psnrs.append(psnr)
        self.episode_bitrates.append(bitrate)
        self.episode_ssims.append(ssim)
        self.episode_rewards.append(reward)
        
        return reward

    def _get_next_observation(self):
        """
        Get the observation for the next block or frame.
        
        In practice, this would get the actual features of the next block from the encoder.
        
        Returns:
            observation: Feature vector for the next block
        """
        # Placeholder for demonstration
        # In real implementation, this would extract features from the actual block
        observation = np.random.rand(len(self.config['features'])).astype(np.float32)
        return observation

    def _get_next_frame_blocks(self):
        """
        Get the blocks for the next frame.
        
        In practice, this would get the actual blocks from the encoder for the next frame.
        
        Returns:
            blocks: List of blocks in the next frame
        """
        # Placeholder for demonstration
        # In real implementation, this would get actual blocks from the encoder
        num_blocks = 10  # Example number of blocks
        return [i for i in range(num_blocks)]

    def render(self):
        """
        Render the current state of the environment.
        
        Currently only supports 'human' mode which prints statistics.
        """
        if self.render_mode == "human":
            if self.episode_rewards:
                avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
                avg_psnr = sum(self.episode_psnrs) / len(self.episode_psnrs)
                avg_bitrate = sum(self.episode_bitrates) / len(self.episode_bitrates)
                avg_ssim = sum(self.episode_ssims) / len(self.episode_ssims)
                
                print(f"Episode: {self.current_episode}")
                print(f"Frame: {self.frame_idx}, Block: {self.current_block_idx}")
                print(f"Avg Reward: {avg_reward:.4f}")
                print(f"Avg PSNR: {avg_psnr:.2f} dB")
                print(f"Avg Bitrate: {avg_bitrate:.6f} bpp")
                print(f"Avg SSIM: {avg_ssim:.4f}")
                print("-" * 40)

    def close(self):
        """
        Clean up resources when environment is no longer needed.
        """
        self.logger.info("Environment closed")
        super().close()


class SVTAVIEncoderEnv(VideoEncoderEnv):
    """
    Specific implementation of VideoEncoderEnv for the SVT-AV1 encoder.
    
    This class extends the base VideoEncoderEnv to interact specifically with
    the SVT-AV1 encoder through the provided Python hooks.
    """
    
    def __init__(self, 
                 config: Dict[str, Any] = None,
                 render_mode: Optional[str] = None):
        """
        Initialize the SVT-AV1 encoder environment.
        
        Args:
            config: Configuration dictionary for the environment
            render_mode: Rendering mode ('human' or None)
        """
        super().__init__(config, render_mode)
        
        # SVT-AV1 specific state tracking
        self.frame_feedback_buffer = {}  # Store frame feedback by picture_number
        self.sb_state_buffer = {}  # Store superblock states
        self.sb_action_buffer = {}  # Store superblock actions
        self.frame_sbs = {}  # Group superblocks by frame
        
        self.logger.info("SVTAVIEncoderEnv initialized")
    
    def process_sb_request(self, sb_info):
        """
        Process a superblock request from the encoder hook.
        
        This method is called when the encoder requests a QP offset for a superblock.
        It receives the superblock information, selects an action using the current policy,
        and returns the corresponding QP offset.
        
        Args:
            sb_info: Dictionary containing superblock information from the encoder
            
        Returns:
            qp_offset: Selected QP offset for the superblock
        """
        # Extract sb_index and other relevant information
        sb_index = sb_info.get('sb_index')
        frame_idx = sb_info.get('picture_number', 0)  # If available, otherwise use 0
        
        # Convert superblock information to state representation
        state = self._convert_sb_info_to_state(sb_info)
        
        # Store state for later training
        if frame_idx not in self.frame_sbs:
            self.frame_sbs[frame_idx] = []
        self.frame_sbs[frame_idx].append(sb_index)
        self.sb_state_buffer[f"{frame_idx}_{sb_index}"] = state
        
        # In a real implementation, this would use the current policy to select an action
        # For now, we'll use a placeholder
        action = self.action_space.sample()  # Random action for demonstration
        
        # Store action for later training
        self.sb_action_buffer[f"{frame_idx}_{sb_index}"] = action
        
        # Convert action index to actual QP offset
        qp_offset = action + self.qp_min_offset
        
        self.logger.debug(f"Superblock {sb_index} in frame {frame_idx}: action={action}, qp_offset={qp_offset}")
        
        return qp_offset
    
    def process_frame_feedback(self, frame_feedback):
        """
        Process frame feedback from the encoder hook.
        
        This method is called when the encoder provides feedback after encoding a frame.
        It receives the frame statistics, calculates rewards, and updates the agent.
        
        Args:
            frame_feedback: Dictionary containing frame feedback from the encoder
        """
        # Extract frame information
        picture_number = frame_feedback.get('picture_number')
        psnr_y = frame_feedback.get('luma_psnr', 0)
        psnr_u = frame_feedback.get('cb_psnr', 0)
        psnr_v = frame_feedback.get('cr_psnr', 0)
        ssim_y = frame_feedback.get('luma_ssim', 0)
        ssim_u = frame_feedback.get('cb_ssim', 0)
        ssim_v = frame_feedback.get('cr_ssim', 0)
        bitrate = frame_feedback.get('picture_stream_size', 0)
        
        # Store frame feedback for later use
        self.frame_feedback_buffer[picture_number] = frame_feedback
        
        # Calculate average PSNR and SSIM
        avg_psnr = (psnr_y + psnr_u + psnr_v) / 3.0
        avg_ssim = (ssim_y + ssim_u + ssim_v) / 3.0
        
        # Log frame statistics
        self.logger.info(f"Frame {picture_number}: PSNR={avg_psnr:.2f}dB, SSIM={avg_ssim:.4f}, Bitrate={bitrate} bytes")
        
        # Calculate and assign rewards to all superblocks in this frame
        self._calculate_sb_rewards(picture_number, avg_psnr, avg_ssim, bitrate)
        
        # In a real implementation, this would trigger training of the RL agent
        # based on the collected experience
        
    def _convert_sb_info_to_state(self, sb_info):
        """
        Convert superblock information from the encoder to state representation.
        
        Args:
            sb_info: Dictionary containing superblock information from the encoder
            
        Returns:
            state: Numpy array containing the state representation
        """
        # Extract features according to the defined feature list
        features = []
        for feature in self.config['features']:
            if feature in sb_info:
                features.append(float(sb_info[feature]))
            else:
                features.append(0.0)  # Default value if feature is missing
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_sb_rewards(self, picture_number, avg_psnr, avg_ssim, bitrate):
        """
        Calculate rewards for all superblocks in a frame.
        
        Args:
            picture_number: Frame index
            avg_psnr: Average PSNR for the frame
            avg_ssim: Average SSIM for the frame
            bitrate: Frame bitrate in bytes
        """
        if picture_number not in self.frame_sbs:
            self.logger.warning(f"No superblocks found for frame {picture_number}")
            return
        
        # Get all superblocks in this frame
        sb_indices = self.frame_sbs[picture_number]
        num_sbs = len(sb_indices)
        
        # Calculate normalized bitrate (bits per superblock)
        bitrate_per_sb = bitrate * 8 / num_sbs
        
        # Calculate rewards for each superblock
        rewards = []
        for sb_index in sb_indices:
            # For now, we'll assign the same reward to all superblocks in the frame
            # In a more sophisticated implementation, rewards could be distributed
            # based on superblock-specific metrics
            
            weights = self.config['reward_weights']
            reward = (
                weights.get('psnr', 1.0) * avg_psnr +
                weights.get('bitrate', -0.5) * bitrate_per_sb +
                weights.get('ssim', 0.3) * avg_ssim
            )
            
            # Store reward for training
            sb_key = f"{picture_number}_{sb_index}"
            if sb_key in self.sb_state_buffer and sb_key in self.sb_action_buffer:
                # In a real implementation, this would be used for training
                state = self.sb_state_buffer[sb_key]
                action = self.sb_action_buffer[sb_key]
                
                # Record reward
                rewards.append(reward)
                
                self.logger.debug(f"SB {sb_index} in frame {picture_number}: reward={reward:.4f}")
        
        # Log average reward for the frame
        if rewards:
            avg_reward = sum(rewards) / len(rewards)
            self.logger.info(f"Frame {picture_number}: Average reward={avg_reward:.4f}")


class DummyEnv(VideoEncoderEnv):
    """
    A dummy environment for testing purposes that doesn't require actual encoder integration.
    This can be used for development and testing of RL algorithms before integrating
    with the actual SVT-AV1 encoder.
    """
    
    def __init__(self, 
                 config: Dict[str, Any] = None,
                 render_mode: Optional[str] = None):
        """
        Initialize the dummy encoder environment.
        
        Args:
            config: Configuration dictionary for the environment
            render_mode: Rendering mode ('human' or None)
        """
        # Default configuration for dummy environment
        dummy_config = {
            'frame_width': 352,
            'frame_height': 288,
            'total_frames': 100,
            'blocks_per_frame': 100
        }
        
        # Update with provided config
        if config:
            dummy_config.update(config)
        
        super().__init__(dummy_config, render_mode)
        
        # Set total frames for the episode
        self.total_frames = self.config['total_frames']
        
        self.logger.info(f"DummyEnv initialized with {self.total_frames} frames")
    
    def reset(self, *, seed=None, options=None):
        """
        Reset the environment at the start of a new episode.
        
        Args:
            seed: Random seed
            options: Additional options for reset
            
        Returns:
            Tuple of (observation, info)
        """
        observation, info = super().reset(seed=seed, options=options)
        
        # Create blocks for the first frame
        self.blocks_in_frame = self._get_next_frame_blocks()
        
        # Get initial observation
        observation = self._get_next_observation()
        
        return observation, info
    
    def _get_next_frame_blocks(self):
        """
        Generate dummy blocks for the next frame.
        
        Returns:
            List of block indices
        """
        return list(range(self.config['blocks_per_frame']))
    
    def _get_next_observation(self):
        """
        Generate observation for the current block.
        
        Returns:
            Numpy array with observation features
        """
        # For testing purposes, create a simple state representation
        # In a real implementation, this would contain meaningful features
        
        # Simple features:
        # 1. Normalized block position (x, y) in [0, 1]
        # 2. Normalized frame index in [0, 1]
        # 3. Random texture complexity in [0, 1]
        # 4. Random motion magnitude in [0, 1]
        # ... and so on for all features
        
        block_x = (self.current_block_idx % 10) / 10.0  # Simple grid layout
        block_y = (self.current_block_idx // 10) / 10.0
        frame_progress = self.frame_idx / self.total_frames
        
        # Create observation with appropriate length
        observation = np.zeros(len(self.config['features']), dtype=np.float32)
        
        # Fill in some basic features
        observation[0] = self.current_block_idx  # sb_index
        observation[1] = block_x * self.config['frame_width']  # sb_origin_x
        observation[2] = block_y * self.config['frame_height']  # sb_origin_y
        
        # Add some random features for texture complexity, motion, etc.
        for i in range(3, len(observation)):
            observation[i] = np.random.rand()
        
        return observation
    
    def _calculate_reward(self, qp_offset):
        """
        Calculate reward for the current action.
        
        Args:
            qp_offset: The QP offset action
            
        Returns:
            reward: Calculated reward value
        """
        # Simple reward model for testing:
        # - Small QP offsets are generally good (close to 0)
        # - Very large positive/negative offsets are penalized
        # - Some content/motion-dependent variation
        
        # Base reward depends on how close the offset is to 0
        base_reward = -0.5 * abs(qp_offset)
        
        # Add some content-dependent variation
        block_complexity = np.random.rand()  # Random complexity for testing
        content_factor = block_complexity * qp_offset
        
        # Lower QP (negative offset) helps with complex content
        # Higher QP (positive offset) is okay for simple content
        content_reward = -content_factor
        
        # Total reward
        reward = base_reward + content_reward
        
        # Record statistics
        self.episode_rewards.append(reward)
        
        # Simulate PSNR (inversely related to QP)
        psnr = 35 - 0.5 * qp_offset + np.random.normal(0, 0.5)
        self.episode_psnrs.append(psnr)
        
        # Simulate bitrate (inversely related to QP)
        bitrate = 0.1 * (1.0 - 0.05 * qp_offset) + np.random.normal(0, 0.01)
        self.episode_bitrates.append(bitrate)
        
        # Simulate SSIM (inversely related to QP)
        ssim = 0.95 - 0.01 * qp_offset + np.random.normal(0, 0.005)
        self.episode_ssims.append(ssim)
        
        return reward


# Example usage
if __name__ == "__main__":
    # Create and test the dummy environment
    env = DummyEnv(render_mode="human")
    
    observation, info = env.reset()
    
    total_reward = 0
    done = False
    
    for _ in range(100):
        action = env.action_space.sample()  # Random action
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            observation, info = env.reset()
            print(f"Episode completed. Total reward: {total_reward}")
            total_reward = 0
        
        time.sleep(0.01)  # Slow down for visualization
    
    env.close()