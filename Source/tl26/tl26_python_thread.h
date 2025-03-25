#ifndef TL26_PYTHON_THREAD_H
#define TL26_PYTHON_THREAD_H

#include <Python.h>
#include <pthread.h>
#include <stdbool.h>
#include "../Lib/Codec/pic_buffer_desc.h"



typedef enum {
    PY_REQUEST_FRAME_FEEDBACK,
    PY_REQUEST_SB_FEEDBACK,
    PY_REQUEST_SB_OFFSET
} PyRequestType;


typedef struct {
    PyRequestType type;
    
    union {
        struct {
            int picture_number;
            int temporal_layer_index;
            int qp;
            int avg_qp;
            double luma_psnr;
            double cb_psnr;
            double cr_psnr;
            double mse_y;
            double mse_u;
            double mse_v;
            double luma_ssim;
            double cb_ssim;
            double cr_ssim;
            int picture_stream_size;
        } frame_feedback;

        struct {
            int picture_number;
            unsigned sb_index;
            unsigned sb_origin_x;
            unsigned sb_origin_y;
            double luma_psnr;
            double cb_psnr;
            double cr_psnr;
            double mse_y;
            double mse_u;
            double mse_v;
            double luma_ssim;
            double cb_ssim;
            double cr_ssim;
            uint8_t* buffer_y;
            uint8_t* buffer_cb;
            uint8_t* buffer_cr;
            u_int16_t sb_width;
            u_int16_t sb_height;
        } sb_feedback;
        
        struct {
            unsigned sb_index;
            unsigned sb_origin_x;
            unsigned sb_origin_y;
            int sb_qp;
            int sb_final_blk_cnt;
            int mi_row_start;
            int mi_row_end;
            int mi_col_start;
            int mi_col_end;
            int tg_horz_boundary;
            int tile_row;
            int tile_col;
            int tile_rs_index;
            uint8_t *buffer_y;
            uint8_t *buffer_cb;
            uint8_t *buffer_cr;
            u_int16_t sb_width;
            u_int16_t sb_height;
            int picture_number;
            int encoder_bit_depth;
            int qindex;
            double beta;
            int type;
        } sb_offset;
    } params;
    
    union {
        int int_result;
        void* ptr_result;
    } result;
    
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    bool completed;
} PyRequest;

typedef struct {
    PyRequest* requests;
    int capacity;
    int head;
    int tail;
    int size;
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
} PyRequestQueue;


typedef struct {
    pthread_t thread_id;
    PyRequestQueue queue;
    bool running;
    PyObject* frame_feedback_func;
    PyObject* sb_feedback_func;
    PyObject* sb_offset_func;
} TL26ThreadState;

void init_python_thread(void);
void shutdown_python_thread(void);
int submit_frame_feedback_request(int picture_number, int temporal_layer_index, int qp, int avg_qp,
                                  double luma_psnr, double cb_psnr, double cr_psnr,
                                  double mse_y, double mse_u, double mse_v,
                                  double luma_ssim, double cb_ssim, double cr_ssim,
                                  int picture_stream_size);
int submit_sb_feedback_request(int picture_number, int sb_index, unsigned sb_origin_x, unsigned sb_origin_y,
                               double luma_psnr, double cb_psnr, double cr_psnr,
                               double mse_y, double mse_u, double mse_v,
                               double luma_ssim, double cb_ssim, double cr_ssim,
                               uint8_t* buffer_y, uint8_t* buffer_cb, uint8_t* buffer_cr,
                               u_int16_t sb_width, u_int16_t sb_height);
int submit_sb_offset_request(unsigned sb_index, unsigned sb_origin_x, unsigned sb_origin_y,
                             int sb_qp, int sb_final_blk_cnt, int mi_row_start, int mi_row_end,
                             int mi_col_start, int mi_col_end, int tg_horz_boundary,
                             int tile_row, int tile_col, int tile_rs_index,
                             int picture_number,
                             u_int8_t* buffer_y, u_int8_t* buffer_cb, u_int8_t* buffer_cr,
                             u_int16_t sb_width, u_int16_t sb_height,
                             int encoder_bit_depth, int qindex, double beta, int type);

                            
void cleanup_python_thread_objects(void);
void set_python_thread_running(bool running);
void signal_python_thread_termination(void);
bool is_python_thread_running(void);
#endif // TL26_PYTHON_THREAD_H