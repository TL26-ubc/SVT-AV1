import av1gym.pyencoder as bridge

def get_deltaq_offset(sbs: list[bridge.SuperBlockInfo], frame_type: int, frame_number: int) -> list[int]:
    print("In python: ", len(sbs), frame_type, frame_number)
    for sb in sbs:
        if sb["sb_x_mv"] > 0:
            print("sb_x_mv: ", sb["sb_x_mv"])
        if sb["sb_y_mv"] > 0:
            print("sb_y_mv: ", sb["sb_y_mv"])

    return [0]*len(sbs)

def picture_feedback(a, b, c):
    pass

bridge.register_callbacks(get_deltaq_offset=get_deltaq_offset, picture_feedback=picture_feedback)

args = {
    "input": "../../playground/bus_cif.y4m",
    "pred_struct": 1,
    "rc": 2,
    "tbr": 100,
    "enable_stat_report": True
}

bridge.run(**args)