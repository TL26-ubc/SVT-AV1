import pyencoder, os

# check for --file argument
if "--file" in os.sys.argv:
    file_index = os.sys.argv.index("--file") + 1
    if file_index < len(os.sys.argv):
        input_file = os.sys.argv[file_index]
        # make it absolute path
        input_file = os.path.abspath(input_file)
        print(f"Input file: {input_file}")
    else:
        raise ValueError("No input file provided after --file argument.")
else:
    raise ValueError("No --file argument provided. Provide the video file to encode.")

def get_deltaq_offset(
    sb_index: int, sb_org_x: int, sb_org_y: int, sb_qindex: int, sb_final_blk_cnt: int,
    mi_row_start: int, mi_row_end: int, mi_col_start: int, mi_col_end: int,
    tg_horz_boundary: int, tile_row: int, tile_col: int, tile_rs_index: int,
    encoder_bit_depth: int, beta: float, is_intra: bool,
) -> int:
    print(
        "In python:", sb_index, sb_org_x, sb_org_y, sb_qindex, sb_final_blk_cnt,
        mi_row_start, mi_row_end, mi_col_start, mi_col_end, 
        tg_horz_boundary, tile_row, tile_col, tile_rs_index,
        encoder_bit_depth, beta, is_intra,
    )
    return 32

pyencoder.register_callbacks(get_deltaq_offset=get_deltaq_offset)
pyencoder.run(input=input_file, rc=True, enable_stat_report=True)