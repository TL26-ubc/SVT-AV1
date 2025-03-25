from tl26.train import train


class TileInfo:
    def __init__(self, mi_row_start: int, mi_row_end: int, 
                 mi_col_start: int, mi_col_end: int, 
                 tg_horz_boundary: int, 
                 tile_row: int, tile_col: int, tile_rs_index: int):
        self.mi_row_start = mi_row_start
        self.mi_row_end = mi_row_end
        self.mi_col_start = mi_col_start
        self.mi_col_end = mi_col_end
        self.tg_horz_boundary = tg_horz_boundary
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.tile_rs_index = tile_rs_index
        
    def to_float_list(self): # adjust as needed
        return [float(self.mi_row_start), float(self.mi_row_end), float(self.mi_col_start), float(self.mi_col_end),
                float(self.tg_horz_boundary), float(self.tile_row), float(self.tile_col), float(self.tile_rs_index)]

class SuperBlock:
    def __init__(self, index: int, org_x: int, org_y:int, qindex: int, final_blk_cnt: int,
                #  pcs: PictureControlSet,
                # av1xd: MacroBlockD,
                 tile_info: TileInfo
                 ):
        self.index = index
        self.org_x = org_x
        self.org_y = org_y
        self.qindex = qindex
        self.final_blk_cnt = final_blk_cnt
        # self.pcs = pcs
        # self.av1xd = av1xd
        self.tile_info = tile_info
        
    def to_float_list(self): # adjust as needed
        return [float(self.index), float(self.org_x), float(self.org_y), float(self.qindex), float(self.final_blk_cnt)] + self.tile_info.to_float_list()


class Reqeust_sb_offset:
    def __init__(self, superblock: SuperBlock,
                 picture_number: int,
                 encoder_bit_depth: int, qindex: int,
                 beta: float,
                 slice_type_is_I_SLICE: bool
                 ):
        self.superblock = superblock
        self.picture_number = picture_number
        self.encoder_bit_depth = encoder_bit_depth
        self.qindex = qindex
        self.beta = beta
        self.slice_type_is_I_SLICE = slice_type_is_I_SLICE

    def to_float_list(self):
        return self.superblock.to_float_list() + \
    [float(self.picture_number), float(self.encoder_bit_depth), float(self.qindex), \
     float(self.beta), float(self.slice_type_is_I_SLICE)] 
    
def sb_send_offset_request(
                # Type: SuperBlock
                index: int, org_x: int, org_y:int, qindex: int, final_blk_cnt: int,
                
                # Type: TileInfo
                mi_row_start: int, mi_row_end: int, 
                mi_col_start: int, mi_col_end: int, 
                tg_horz_boundary: int, 
                tile_row: int, tile_col: int, tile_rs_index: int,
                
                # Reqeust_sb_offset
                picture_number: int,
                encoder_bit_depth: int, same_qindex: int,
                beta: float,
                slice_type_is_I_SLICE: bool,
                buffer_y : list,
                buffer_cb : list,
                buffer_cr : list
                           ):
    # TODO: Implement this function
    tile_info = TileInfo(mi_row_start, mi_row_end, mi_col_start, mi_col_end, tg_horz_boundary, tile_row, tile_col, tile_rs_index)
    superblock = SuperBlock(index, org_x, org_y, qindex, final_blk_cnt, tile_info)
    request = Reqeust_sb_offset(superblock, picture_number, encoder_bit_depth, same_qindex, beta, slice_type_is_I_SLICE)
    print(f"Requesting SB offset from frame {picture_number}")
    print(request.to_float_list())
    print(f"Buffer Y shape: ({len(buffer_y)}, {len(buffer_y[0]) if buffer_y else 0})")
    print(f"Buffer Cb shape: ({len(buffer_cb)}, {len(buffer_cb[0]) if buffer_cb else 0})")
    print(f"Buffer Cr shape: ({len(buffer_cr)}, {len(buffer_cr[0]) if buffer_cr else 0})")
    
    done = False
    state = superblock.to_float_list()
    next_state  = state # TODO: implement next state
    # return train(state)
    return 0