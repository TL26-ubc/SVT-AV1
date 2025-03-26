#ifndef TL26_PYTHON_THREAD_H
#define TL26_PYTHON_THREAD_H

#include <Python.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdatomic.h>
#include "../Lib/Codec/pic_buffer_desc.h"

typedef struct {
    bool      running;
    PyObject* frame_feedback_func;
    PyObject* sb_feedback_func;
    PyObject* sb_offset_func;
} TL26ThreadState;

extern TL26ThreadState py_thread_state;

void init_python_thread(void);
void shutdown_python_thread(void);
void set_python_thread_running(bool running);
bool is_python_thread_running(void);
void signal_python_thread_termination(void);

int submit_frame_feedback_request(int picture_number, int temporal_layer_index, int qp, int avg_qp, double luma_psnr,
                                  double cb_psnr, double cr_psnr, double mse_y, double mse_u, double mse_v,
                                  double luma_ssim, double cb_ssim, double cr_ssim, int picture_stream_size);

int submit_sb_feedback_request(int picture_number, int sb_index, unsigned sb_origin_x, unsigned sb_origin_y,
                               double luma_psnr, double cb_psnr, double cr_psnr, double mse_y, double mse_u,
                               double mse_v, double luma_ssim, double cb_ssim, double cr_ssim, uint8_t* buffer_y,
                               uint8_t* buffer_cb, uint8_t* buffer_cr, uint16_t sb_width, uint16_t sb_height);

int submit_sb_offset_request(unsigned sb_index, unsigned sb_origin_x, unsigned sb_origin_y, int sb_qp,
                             int sb_final_blk_cnt, int mi_row_start, int mi_row_end, int mi_col_start, int mi_col_end,
                             int tg_horz_boundary, int tile_row, int tile_col, int tile_rs_index, int picture_number,
                             uint8_t* buffer_y, uint8_t* buffer_cb, uint8_t* buffer_cr, uint16_t sb_width,
                             uint16_t sb_height, int encoder_bit_depth, int qindex, double beta, int type);

#endif // TL26_PYTHON_THREAD_H