from .feedback import *

frame_feed_backs = {} # type: Dict[int, Frame_feedback] # key: picture_number
sb_feed_backs = {} # type: Dict[(int, int), Frame_feedback] # key: picture_number, sb_index