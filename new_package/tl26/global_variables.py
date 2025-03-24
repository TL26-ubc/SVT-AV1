from __future__ import annotations

from copy import deepcopy
from typing import Dict, Tuple
from tl26.feedback import Frame_feedback, Superblock_feedback

frame_feedbacks = {}  # type: Dict[int, Frame_feedback]  # key: picture_number
sb_feedbacks = {}  # type: Dict[Tuple[int, int], Superblock_feedback]  # key: (picture_number, sb_index)
baseline_frame_feedbacks = {}  # type: Dict[int, Frame_feedback]  # key: picture_number
baseline_sb_feedbacks = {}  # type: Dict[Tuple[int, int], Superblock_feedback]  # key: (picture_number, sb_index)

def get_all_baseline_frame_feedbacks():
    global baseline_frame_feedbacks
    return baseline_frame_feedbacks

def get_all_baseline_sb_feedbacks() -> Dict[Tuple[int, int], Superblock_feedback]:
    global baseline_sb_feedbacks
    return baseline_sb_feedbacks

def get_baseline_frame_feedback(picture_number: int) -> Frame_feedback:
    global baseline_frame_feedbacks
    return baseline_frame_feedbacks[picture_number]

def get_baseline_sb_feedback(picture_number: int, sb_index: int) -> Superblock_feedback:
    global baseline_sb_feedbacks
    return baseline_sb_feedbacks[(picture_number, sb_index)]

def copy_baseline():
    global baseline_frame_feedbacks, baseline_sb_feedbacks, frame_feedbacks, sb_feedbacks
    baseline_frame_feedbacks = deepcopy(frame_feedbacks)
    baseline_sb_feedbacks = deepcopy(sb_feedbacks)
    return

def clear_global_vars():
    global frame_feedbacks, sb_feedbacks
    frame_feedbacks = {}
    sb_feedbacks = {}
    return

def get_all_frame_feedbacks() -> Dict[int, Frame_feedback]:
    global frame_feedbacks
    return frame_feedbacks

def get_all_sb_feedbacks() -> Dict[(int, int), Superblock_feedback]:
    global sb_feedbacks
    return sb_feedbacks

def get_frame_feedback(picture_number: int) -> Frame_feedback:
    global frame_feedbacks
    return frame_feedbacks[picture_number]

def get_sb_feedback(picture_number: int, sb_index: int) -> Superblock_feedback:
    global sb_feedbacks
    return sb_feedbacks[(picture_number, sb_index)]

def add_frame_feedback(frame_feedback: Frame_feedback):
    global frame_feedbacks
    # check no previous feedback
    if frame_feedback.picture_number in frame_feedbacks:
        raise ValueError(f"Frame_feedback already exists for picture_number {frame_feedback.picture_number}")

    frame_feedbacks[frame_feedback.picture_number] = frame_feedback
    return

def add_sb_feedback(sb_feedback: Superblock_feedback):    
    global sb_feedbacks
    # check no previous feedback
    if (sb_feedback.picture_number, sb_feedback.sb_index) in sb_feedbacks:
        raise ValueError(f"Frame_feedback already exists for picture_number {sb_feedback.picture_number}, sb_index {sb_feedback.sb_index}")

    sb_feedbacks[(sb_feedback.picture_number, sb_feedback.sb_index)] = sb_feedback
    sb_feedback.report()
    return