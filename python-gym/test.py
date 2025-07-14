import av1gym.pyencoder as bridge
import random

i = 0
def picture_feedback(buffer_level, b):
    print('python:', buffer_level)
    global i
    i += 1
    pass

bridge.register_callbacks(postencode_feedback=picture_feedback)

args = {
    "input": "../../playground/bus_cif_rev.y4m",
    "output": "../../playground/out.mkv",
    "pred_struct": 1,
    "rc": 2,
    "tbr": 100,
    "enable_stat_report": True
}

bridge.run(**args)

print(i)
