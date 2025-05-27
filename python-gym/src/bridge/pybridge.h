#ifndef PYBRIDGE_H
#define PYBRIDGE_H

#include <Python.h>
#include <stdint.h>
#include <stdbool.h>
#include "../../../Source/API/EbSvtAv1Enc.h"

extern int (*get_deltaq_offset_cb)(SuperBlockInfo *sb_info_array, int *offset_array, uint32_t sb_count,
                                  int32_t picture_number, int32_t frame_type, void *user);

extern void (*recv_frame_feedback_cb)(uint8_t *buffer_y, uint8_t *buffer_cb, uint8_t *buffer_cr, uint32_t picture_number,
                                 uint32_t origin_x, uint32_t origin_y, uint32_t stride_y, uint32_t stride_cb,
                                 uint32_t stride_cr, uint32_t width, uint32_t height, void *user);

int get_deltaq_offset_trampoline(SuperBlockInfo *sb_info_array, int *offset_array, uint32_t sb_count,
                                int32_t picture_number, int32_t frame_type, void *user);

void recv_frame_feedback_trampoline(uint8_t *buffer_y, uint8_t *buffer_cb, uint8_t *buffer_cr, uint32_t picture_number,
                                    uint32_t origin_x, uint32_t origin_y, uint32_t stride_y, uint32_t stride_cb,
                                    uint32_t stride_cr, uint32_t width, uint32_t height, void *user);

#endif /* PYBRIDGE_H */
