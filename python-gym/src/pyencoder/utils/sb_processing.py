from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class SuperBlockInfo:
    sb_org_x: int
    sb_org_y: int
    sb_width: int
    sb_height: int
    sb_qindex: int
    sb_x_mv: int
    sb_y_mv: int
    beta: float

def psnr_block(
    block1 : np.ndarray,
    block2 : np.ndarray
    ) -> float:
    mse = np.mean((block1.astype(np.float32) - block2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255 * 255 / mse)

def per_superblock_psnr(
    frame1 : np.ndarray, 
    frame2 : np.ndarray, 
    block_size=64
    ) -> list[list[float]]:
    """
    Calculates the PSNR (Peak Signal-to-Noise Ratio) for each superblock between two frames.
    Divides the input frames into non-overlapping blocks of size `block_size` x `block_size` and computes the PSNR for each corresponding block. Returns a 2D list (map) of PSNR values, where each element corresponds to a block. If a block is incomplete or mismatched at the frame edges, `None` is inserted for that block.
    Args:
        frame1 (np.ndarray): The first input frame (image) as a NumPy array.
        frame2 (np.ndarray): The second input frame (image) as a NumPy array, must have the same shape as `frame1`.
        block_size (int, optional): The size of the superblock (default is 64).
    Returns:
        list[list[float or None]]: A 2D list containing the PSNR value for each block, or `None` for blocks that could not be compared.
    """
    
    h, w = frame1.shape[:2]
    psnr_map = []
    for y in range(0, h, block_size):
        row = []
        for x in range(0, w, block_size):
            y_end = min(y + block_size, h)
            x_end = min(x + block_size, w)
            block1 = frame1[y:y_end, x:x_end]
            block2 = frame2[y:y_end, x:x_end]
            if block1.shape == block2.shape and block1.size > 0:
                row.append(psnr_block(block1, block2))
            else:
                row.append(None)  # or handle edge blocks as needed
        psnr_map.append(row)
    return psnr_map

def get_frame_psnr(
    original_frame: np.ndarray,
    modified_frame: np.ndarray,
) -> float:
    """
    Calculate the PSNR (Peak Signal-to-Noise Ratio) between two video frames.
    
    Args:
        original_frame (np.ndarray): The original video frame.
        modified_frame (np.ndarray): The modified video frame.
        
    Returns:
        float: The PSNR value between the two frames.
    """
    return cv2.PSNR(original_frame, modified_frame
)


def get_video_psnr(
    original_video: cv2.VideoCapture,
    modified_video: cv2.VideoCapture
) -> list[float]:
    """
    Calculates the PSNR (Peak Signal-to-Noise Ratio) for each pair of corresponding frames in two videos.

    Args:
        original_video (cv2.VideoCapture): VideoCapture object for the original (reference) video.
        modified_video (cv2.VideoCapture): VideoCapture object for the modified (processed) video.

    Returns:
        list[float]: A list of PSNR values, one for each pair of frames.

    Raises:
        ValueError: If the videos do not have the same number of frames.

    Note:
        Assumes that the videos are synchronized and have the same resolution and frame order.
    """
    psnr_list = []
    while True:
        ret1, frame1 = original_video.read()
        ret2, frame2 = modified_video.read()
        if not ret1 and not ret2:
            # Both videos have ended
            break
        if not ret1 or not ret2:
            raise ValueError("Videos must have the same number of frames.")
        psnr_list.append(get_frame_psnr(frame1, frame2))
    return psnr_list
    
def get_frame_state(
    frame: np.ndarray,
    block_size: int = 64
) -> list[list[float]]:
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
    h, w = frame.shape[:2]
    y_comp_list = []
    h_mv_list = []
    v_mv_list = []
    beta_list = []
    
    for y in range(0, h, block_size):
        for x in range(0, w, block_size): # follow encoder order, x changes first
            y_end = min(y + block_size, h)
            x_end = min(x + block_size, w)
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
            
    return [
        y_comp_list,
        h_mv_list,
        v_mv_list,
        beta_list
    ]
    
def get_x_frame_state(
    frame_num: int,
    video_cv2: cv2.VideoCapture,
    block_size: int = 64
) -> list[list[float]]:
    """
    Extracts the state of a specific video frame based on superblock information.
    
    Args:
        frame_num (int): The frame number to extract the state from.
        video_cv2 (cv2.VideoCapture): The VideoCapture object for the video.
        block_size (int): Size of the blocks to be processed. Should be 64 in SVT-AV1.
        
    Returns:
        a list of lists containing superblock information:
            0 Y-component variance of all superblocks in the frame 
            1 Horizontal and 
            2 vertical difference of all superblocks in the frame
            3 Gradient magnitude of all superblocks in the frame
    """
    video_cv2.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = video_cv2.read()
    if not ret:
        raise ValueError(f"Could not read frame {frame_num} from video.")
    
    return get_frame_state(frame, block_size=block_size)
            
            
def get_num_superblock(
    frame_or_video: np.ndarray | cv2.VideoCapture | str,
    block_size: int = 64
) -> int:
    """
    Get the number of superblocks in a video frame or video.

    Args:
        frame_or_video (np.ndarray or cv2.VideoCapture): The video frame or video capture object.
        block_size (int): Size of the blocks to be processed. Should be 64 in SVT-AV1.

    Returns:
        int: Number of superblocks in the frame or in the first frame of the video.
    """
    if isinstance(frame_or_video, np.ndarray):
        h, w = frame_or_video.shape[:2]
    elif isinstance(frame_or_video, cv2.VideoCapture):
        pos = frame_or_video.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = frame_or_video.read()
        if not ret:
            raise ValueError("Could not read frame from video.")
        h, w = frame.shape[:2]
        frame_or_video.set(cv2.CAP_PROP_POS_FRAMES, pos)
    elif isinstance(frame_or_video, str):
        video_cv2 = cv2.VideoCapture(frame_or_video)
        if not video_cv2.isOpened():
            raise ValueError(f"Could not open video file: {frame_or_video}")
        pos = video_cv2.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = video_cv2.read()
        if not ret:
            raise ValueError("Could not read frame from video.")
        h, w = frame.shape[:2]
        video_cv2.set(cv2.CAP_PROP_POS_FRAMES, pos)
        video_cv2.release()
    else:
        raise TypeError("Input must be a numpy.ndarray or cv2.VideoCapture.")
    num_blocks_h = (h + block_size - 1) // block_size
    num_blocks_w = (w + block_size - 1) // block_size
    return num_blocks_h * num_blocks_w
            
            
