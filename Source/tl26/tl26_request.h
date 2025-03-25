#include "../Lib/Codec/coding_unit.h"
#include "../Lib/Codec/block_structures.h"
#include "../Lib/Codec/pcs.h"
int request_sb_offset(SuperBlock *sb_ptr, PictureControlSet *pcs, int encoder_bit_depth, int qindex, double beta, bool slice_type_is_I_SLICE);
