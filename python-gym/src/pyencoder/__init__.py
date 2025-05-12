from importlib import import_module as _imp

_svtapp = _imp("._svtapp", package=__name__)
_run = _svtapp.run
_register = _svtapp.register_callbacks

def run(**kwargs):
    argv = ["svtav1"]
    for key, val in kwargs.items():
        flag = f"--{key.replace('_', '-')}"   # ex: rc_mode -> --rc-mode
        if isinstance(val, bool):             # True/False -> 1/0
            argv.extend([flag, "1" if val else "0"])
        else:
            argv.extend([flag, str(val)])

    print(argv)
    _run(argv)

def register_callbacks(*, get_deltaq_offset=None):
    _register(get_deltaq_offset)