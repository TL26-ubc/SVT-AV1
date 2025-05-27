from importlib import import_module as _imp

_svtapp = _imp("._svtapp", package="pyencoder")
_run = _svtapp.run
_register = _svtapp.register_callbacks


def run(**kwargs):
    argv = ["svtav1"]
    for key, val in kwargs.items():
        # Determine short or long flag
        flag = f"-{key}" if len(key) == 1 else f"--{key.replace('_', '-')}"

        # Convert value to string appropriately
        if isinstance(val, bool):
            argv.extend([flag, "1" if val else "0"])
        else:
            argv.extend([flag, str(val)])

    print(argv)
    _run(argv)


def register_callbacks(*, get_deltaq_offset=None, frame_feedback=None, picture_feedback=None):
    _register(get_deltaq_offset, frame_feedback, picture_feedback)
