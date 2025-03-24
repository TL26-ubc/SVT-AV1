from __future__ import annotations

from copy import deepcopy
from typing import Dict, Tuple

frame_feed_backs = {}  # type: Dict[int, Frame_feedback]  # key: picture_number
sb_feed_backs = {}  # type: Dict[Tuple[int, int], Superblock_feedback]  # key: (picture_number, sb_index)
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
    global baseline_frame_feedbacks, baseline_sb_feedbacks, frame_feed_backs, sb_feed_backs
    baseline_frame_feedbacks = deepcopy(frame_feed_backs)
    baseline_sb_feedbacks = deepcopy(sb_feed_backs)
    return

def clear_global_vars():
    global frame_feed_backs, sb_feed_backs
    frame_feed_backs = {}
    sb_feed_backs = {}
    return

def get_all_frame_feedbacks() -> Dict[int, Frame_feedback]:
    global frame_feed_backs
    return frame_feed_backs

def get_all_sb_feedbacks() -> Dict[(int, int), Superblock_feedback]:
    global sb_feed_backs
    return sb_feed_backs

def get_frame_feedback(picture_number: int) -> Frame_feedback:
    global frame_feed_backs
    return frame_feed_backs[picture_number]

def get_sb_feedback(picture_number: int, sb_index: int) -> Superblock_feedback:
    global sb_feed_backs
    return sb_feed_backs[(picture_number, sb_index)]

def add_frame_feedback(frame_feedback: Frame_feedback):
    global frame_feed_backs
    # check no previous feedback
    if frame_feedback.picture_number in frame_feed_backs:
        raise ValueError(f"Frame_feedback already exists for picture_number {frame_feedback.picture_number}")

    frame_feed_backs[frame_feedback.picture_number] = frame_feedback
    return

def add_sb_feedback(sb_feedback: Superblock_feedback):    
    global sb_feed_backs
    # check no previous feedback
    if (sb_feedback.picture_number, sb_feedback.sb_index) in sb_feed_backs:
        raise ValueError(f"Frame_feedback already exists for picture_number {sb_feedback.picture_number}, sb_index {sb_feedback.sb_index}")

    sb_feed_backs[(sb_feedback.picture_number, sb_feedback.sb_index)] = sb_feedback
    return