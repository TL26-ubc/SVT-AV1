from importlib import import_module as _imp
from typing import TypedDict

_av1_wrapper = _imp("av1gym.pyencoder._av1_wrapper", package=__name__)
_run = _av1_wrapper.run
_register = _av1_wrapper.register_callbacks

class SuperBlockInfo(TypedDict):
    sb_org_x: int
    sb_org_y: int
    sb_width: int
    sb_height: int
    sb_qindex: int
    sb_x_mv: int
    sb_y_mv: int

def run(**kwargs):
    argv = ["svtav1"]
    for key, val in kwargs.items():
        # If key starts with '-', use it directly as a flag
        if key.startswith("-"):
            flag = key
        else:
            # Determine short or long flag
            flag = f"-{key}" if len(key) == 1 else f"--{key.replace('_', '-')}"

        # Convert value to string appropriately
        if isinstance(val, bool):
            argv.extend([flag, "1" if val else "0"])
        else:
            argv.extend([flag, str(val)])

    _run(argv)

def register_callbacks(*, get_deltaq_offset=None, picture_feedback=None, postencode_feedback=None):
    _register(get_deltaq_offset, picture_feedback, postencode_feedback)
