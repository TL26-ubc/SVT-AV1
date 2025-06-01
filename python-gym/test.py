import argparse
import os
from typing import List
import pandas as pd # Import pandas
import json # For potentially serializing complex data

import pyencoder

def get_deltaq_offset(
    sb_info_list: List[dict],
    sb_total_count: int,
    picture_number: int,
    frame_type: int
) -> List[int]:
    return [0]*(sb_total_count-1)

pyencoder.register_callbacks(get_deltaq_offset=get_deltaq_offset)
pyencoder.run(input="../../playground/akiyo_qcif.y4m", rc=True, enable_stat_report=True, tbr=500)
