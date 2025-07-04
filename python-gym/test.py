import pyencoder

def get_deltaq_offset(sbs: list[pyencoder.SuperBlockInfo], frame_type: int, frame_number: int) -> list[int]:
    print("In python: ", len(sbs), frame_type, frame_number)
    return [0]*len(sbs)

def picture_feedback(a, b, c):
    pass

pyencoder.register_callbacks(get_deltaq_offset=get_deltaq_offset, picture_feedback=picture_feedback)

args = {
    "input": "../../playground/bus_cif.y4m",
    "pred_struct": 1,
    "rc": 2,
    "tbr": 100,
    "enable_stat_report": True
}

pyencoder.run(**args)