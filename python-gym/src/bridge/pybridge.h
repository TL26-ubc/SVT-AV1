#ifndef PYBRIDGE_H
#define PYBRIDGE_H

#include <Python.h>
#include <stdint.h>
#include <stdbool.h>

extern int (*get_deltaq_offset_cb)(unsigned sb_index, unsigned sb_org_x, unsigned sb_org_y, uint8_t sb_qindex,
                                   uint16_t sb_final_blk_cnt, int32_t mi_row_start, int32_t mi_row_end,
                                   int32_t mi_col_start, int32_t mi_col_end, int32_t tg_horz_boundary, int32_t tile_row,
                                   int32_t tile_col, int32_t tile_rs_index, int32_t picture_number, uint8_t *buffer_y,
                                   uint8_t *buffer_cb, uint8_t *buffer_cr, uint16_t sb_width, uint16_t sb_height,
                                   uint8_t encoder_bit_depth, int32_t qindex, double beta, int32_t type, void *user);

extern void (*recv_frame_feedback_cb)(uint8_t *buffer_y, uint8_t *buffer_cb, uint8_t *buffer_cr,
                                      uint32_t picture_number, u_int32_t bytes_used, uint32_t origin_x,
                                      uint32_t origin_y, uint32_t stride_y, uint32_t stride_cb, uint32_t stride_cr,
                                      uint32_t width, uint32_t height, void *user);

extern void (*recv_picture_feedback)(uint8_t *bitStream, uint32_t bitstream_size, uint32_t picture_number);

int get_deltaq_offset_trampoline(unsigned sb_index, unsigned sb_org_x, unsigned sb_org_y, uint8_t sb_qindex,
                                 uint16_t sb_final_blk_cnt, int32_t mi_row_start, int32_t mi_row_end,
                                 int32_t mi_col_start, int32_t mi_col_end, int32_t tg_horz_boundary, int32_t tile_row,
                                 int32_t tile_col, int32_t tile_rs_index, int32_t picture_number, uint8_t *buffer_y,
                                 uint8_t *buffer_cb, uint8_t *buffer_cr, uint16_t sb_width, uint16_t sb_height,
                                 uint8_t encoder_bit_depth, int32_t qindex, double beta, int32_t type, void *user);

void recv_frame_feedback_trampoline(uint8_t *buffer_y, uint8_t *buffer_cb, uint8_t *buffer_cr, uint32_t picture_number,
                                    u_int32_t bytes_used, uint32_t origin_x, uint32_t origin_y, uint32_t stride_y,
                                    uint32_t stride_cb, uint32_t stride_cr, uint32_t width, uint32_t height,
                                    void *user);

void recv_picture_feedback_trampoline(uint8_t *bitStream, uint32_t bitstream_size, uint32_t picture_number);

#endif /* PYBRIDGE_H */
