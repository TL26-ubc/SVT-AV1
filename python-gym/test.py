import av1gym.pyencoder as bridge

ally = 0
cnt = 0
def get_deltaq_offset(sbs: list[bridge.SuperBlockInfo], frame_type: int, frame_number: int) -> list[int]:
    # print("In python: ", len(sbs), frame_type, frame_number)
    global ally, cnt
    ally += sum([sb["sb_x_mv"] for sb in sbs])
    cnt += len([sb["sb_x_mv"] for sb in sbs])
    return [0]*len(sbs)

def picture_feedback(a, b, c):
    pass

bridge.register_callbacks(get_deltaq_offset=get_deltaq_offset, picture_feedback=picture_feedback)

args = {
    "input": "../../playground/bus_cif_rev.y4m",
    "output": "../../playground/out.mkv",
    "pred_struct": 1,
    "rc": 2,
    "tbr": 400,
    "enable_stat_report": True
}

bridge.run(**args)

print(ally/cnt)