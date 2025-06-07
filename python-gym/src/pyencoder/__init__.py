from importlib import import_module as _imp

from pyencoder.av1_wrapper import register_callbacks, run

# av1_wrapper = _imp(".av1_wrapper", package=__name__)
_run = run
_register = register_callbacks


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

    print(argv)
    _run(argv)


def register_callbacks(*, get_deltaq_offset=None, picture_feedback=None):
    _register(get_deltaq_offset, picture_feedback)
