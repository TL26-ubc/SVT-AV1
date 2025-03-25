#include "tl26_python_thread.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

//////////////////////////////////////// Helper Functions ////////////////////////////////////////
PyObject *pixel_to_Py_list(uint8_t *buffer, int width, int height) {
    PyObject *list = PyList_New(height);
    for (int i = 0; i < height; i++) {
        PyObject *row = PyList_New(width);
        for (int j = 0; j < width; j++) { PyList_SetItem(row, j, PyLong_FromLong(buffer[i * width + j])); }
        PyList_SetItem(list, i, row);
    }
    return list;
}

void destroy_Py_list(PyObject *list) {
    for (int i = 0; i < PyList_Size(list); i++) { Py_DECREF(PyList_GetItem(list, i)); }
    Py_DECREF(list);
}
//////////////////////////////////////// Helper Functions ////////////////////////////////////////


static TL26ThreadState py_thread_state = {0};

static void init_request_queue(PyRequestQueue* queue, int capacity) {
    queue->requests = (PyRequest*)malloc(capacity * sizeof(PyRequest));
    queue->capacity = capacity;
    queue->head = 0;
    queue->tail = 0;
    queue->size = 0;
    pthread_mutex_init(&queue->mutex, NULL);
    pthread_cond_init(&queue->not_empty, NULL);
    pthread_cond_init(&queue->not_full, NULL);
}

static void destroy_request_queue(PyRequestQueue* queue) {
    if (queue->requests) {
        free(queue->requests);
        queue->requests = NULL;
    }
    pthread_mutex_destroy(&queue->mutex);
    pthread_cond_destroy(&queue->not_empty);
    pthread_cond_destroy(&queue->not_full);
}

static PyRequest* enqueue_request(PyRequestQueue* queue) {
    PyRequest* request = NULL;
    
    pthread_mutex_lock(&queue->mutex);
    
    while (queue->size == queue->capacity && py_thread_state.running) {
        pthread_cond_wait(&queue->not_full, &queue->mutex);
    }
    
    if (!py_thread_state.running) {
        pthread_mutex_unlock(&queue->mutex);
        return NULL;
    }
    
    request = &queue->requests[queue->tail];
    queue->tail = (queue->tail + 1) % queue->capacity;
    queue->size++;
    
    request->completed = false;
    pthread_mutex_init(&request->mutex, NULL);
    pthread_cond_init(&request->cond, NULL);
    
    pthread_cond_signal(&queue->not_empty);
    pthread_mutex_unlock(&queue->mutex);
    
    return request;
}

static PyRequest* dequeue_request(PyRequestQueue* queue) {
    PyRequest* request = NULL;
    
    pthread_mutex_lock(&queue->mutex);
    
    while (queue->size == 0 && py_thread_state.running) {
        pthread_cond_wait(&queue->not_empty, &queue->mutex);
    }
    
    if (queue->size == 0) {
        pthread_mutex_unlock(&queue->mutex);
        return NULL;
    }
    
    request = &queue->requests[queue->head];
    queue->head = (queue->head + 1) % queue->capacity;
    queue->size--;
    
    pthread_cond_signal(&queue->not_full);
    pthread_mutex_unlock(&queue->mutex);
    
    return request;
}

static void process_frame_feedback_request(PyRequest* request) {
    if (py_thread_state.frame_feedback_func && PyCallable_Check(py_thread_state.frame_feedback_func)) {
        PyObject* args = Py_BuildValue(
            "iiiidddddddddi",
            request->params.frame_feedback.picture_number,
            request->params.frame_feedback.temporal_layer_index,
            request->params.frame_feedback.qp,
            request->params.frame_feedback.avg_qp,
            request->params.frame_feedback.luma_psnr,
            request->params.frame_feedback.cb_psnr,
            request->params.frame_feedback.cr_psnr,
            request->params.frame_feedback.mse_y,
            request->params.frame_feedback.mse_u,
            request->params.frame_feedback.mse_v,
            request->params.frame_feedback.luma_ssim,
            request->params.frame_feedback.cb_ssim,
            request->params.frame_feedback.cr_ssim,
            request->params.frame_feedback.picture_stream_size);
        
        PyObject* result = PyObject_CallObject(py_thread_state.frame_feedback_func, args);
        Py_XDECREF(result);
        Py_DECREF(args);
    }
}

static void process_sb_feedback_request(PyRequest* request) {
    PyObject* buffer_y = pixel_to_Py_list(request->params.sb_feedback.buffer_y, 
        request->params.sb_feedback.sb_width, request->params.sb_feedback.sb_height);
    PyObject* buffer_cb = pixel_to_Py_list(request->params.sb_feedback.buffer_cb, 
        request->params.sb_feedback.sb_width / 2, request->params.sb_feedback.sb_height / 2);
    PyObject* buffer_cr = pixel_to_Py_list(request->params.sb_feedback.buffer_cr, 
        request->params.sb_feedback.sb_width / 2, request->params.sb_feedback.sb_height / 2);
    if (py_thread_state.sb_feedback_func && PyCallable_Check(py_thread_state.sb_feedback_func)) {
        PyObject* args = Py_BuildValue(
            "iIIIdddddddddOOO",
            request->params.sb_feedback.picture_number,
            request->params.sb_feedback.sb_index,
            request->params.sb_feedback.sb_origin_x,
            request->params.sb_feedback.sb_origin_y,
            request->params.sb_feedback.luma_psnr,
            request->params.sb_feedback.cb_psnr,
            request->params.sb_feedback.cr_psnr,
            request->params.sb_feedback.mse_y,
            request->params.sb_feedback.mse_u,
            request->params.sb_feedback.mse_v,
            request->params.sb_feedback.luma_ssim,
            request->params.sb_feedback.cb_ssim,
            request->params.sb_feedback.cr_ssim,
            buffer_y,
            buffer_cb,
            buffer_cr);
        
        PyObject* result = PyObject_CallObject(py_thread_state.sb_feedback_func, args);
        Py_XDECREF(result);
        Py_DECREF(args);
        destroy_Py_list(buffer_y);
        destroy_Py_list(buffer_cb);
        destroy_Py_list(buffer_cr);
    }
}

static void process_sb_offset_request(PyRequest* request) {
    PyObject* buffer_y = pixel_to_Py_list(request->params.sb_offset.buffer_y, 
        request->params.sb_offset.sb_width, request->params.sb_offset.sb_height);
    PyObject* buffer_cb = pixel_to_Py_list(request->params.sb_offset.buffer_cb, 
        request->params.sb_offset.sb_width / 2, request->params.sb_offset.sb_height / 2);
    PyObject* buffer_cr = pixel_to_Py_list(request->params.sb_offset.buffer_cr, 
        request->params.sb_offset.sb_width / 2, request->params.sb_offset.sb_height / 2);
    if (py_thread_state.sb_offset_func && PyCallable_Check(py_thread_state.sb_offset_func)) {
        PyObject* args = Py_BuildValue(
            "IIIiiiiiiiiiiiiidiOOO",
            request->params.sb_offset.sb_index,
            request->params.sb_offset.sb_origin_x,
            request->params.sb_offset.sb_origin_y,
            request->params.sb_offset.sb_qp,
            request->params.sb_offset.sb_final_blk_cnt,
            request->params.sb_offset.mi_row_start,
            request->params.sb_offset.mi_row_end,
            request->params.sb_offset.mi_col_start,
            request->params.sb_offset.mi_col_end,
            request->params.sb_offset.tg_horz_boundary,
            request->params.sb_offset.tile_row,
            request->params.sb_offset.tile_col,
            request->params.sb_offset.tile_rs_index,
            request->params.sb_offset.picture_number,
            request->params.sb_offset.encoder_bit_depth,
            request->params.sb_offset.qindex,
            request->params.sb_offset.beta,
            request->params.sb_offset.type,
            buffer_y,
            buffer_cb,
            buffer_cr
        );
        
        PyObject* result = PyObject_CallObject(py_thread_state.sb_offset_func, args);
        if (result != NULL) {
            request->result.int_result = PyLong_AsLong(result);
            Py_DECREF(result);
        } else {
            request->result.int_result = -1;
            PyErr_Print();
        }
        Py_DECREF(args);
        destroy_Py_list(buffer_y);
        destroy_Py_list(buffer_cb);
        destroy_Py_list(buffer_cr);
    } else {
        request->result.int_result = -1;
    }
}

static void process_request(PyRequest* request) {
    static int sb_framefeedback_count = 0;
    static int sb_offset_count = 0;
    static int sb_feedback_count = 0;

    switch (request->type) {
        case PY_REQUEST_FRAME_FEEDBACK:
            sb_framefeedback_count++;
            process_frame_feedback_request(request);
        break;
        case PY_REQUEST_SB_FEEDBACK:
            sb_feedback_count++;
            process_sb_feedback_request(request);
            break;
        case PY_REQUEST_SB_OFFSET:
            sb_offset_count++;
            process_sb_offset_request(request);
            break;
    }
    
    pthread_mutex_lock(&request->mutex);
    request->completed = true;
    pthread_cond_signal(&request->cond);
    pthread_mutex_unlock(&request->mutex);
}

static void* python_thread_func(void* arg) {
    (void)arg;
    
    PyGILState_STATE gstate = PyGILState_Ensure();
    
    PyObject* feedback_module = PyImport_ImportModule("tl26.feedback");
    if (feedback_module) {
        py_thread_state.frame_feedback_func = PyObject_GetAttrString(feedback_module, "frame_report_feedback");
        py_thread_state.sb_feedback_func = PyObject_GetAttrString(feedback_module, "sb_report_feedback");
        Py_DECREF(feedback_module);
    } else {
        PyErr_Print();
    }
    
    PyObject* request_module = PyImport_ImportModule("tl26.request");
    if (request_module) {
        py_thread_state.sb_offset_func = PyObject_GetAttrString(request_module, "sb_send_offset_request");
        Py_DECREF(request_module);
    } else {
        PyErr_Print();
    }
    
    while (py_thread_state.running) {
        PyRequest* request = dequeue_request(&py_thread_state.queue);
        if (request) {
            process_request(request);
        }
    }
    
    Py_XDECREF(py_thread_state.frame_feedback_func);
    Py_XDECREF(py_thread_state.sb_feedback_func);
    Py_XDECREF(py_thread_state.sb_offset_func);
    
    PyGILState_Release(gstate);
    
    return NULL;
}

void init_python_thread(void) {
    init_request_queue(&py_thread_state.queue, 100);
    py_thread_state.running = true;
    
    if (!Py_IsInitialized()) {
        Py_Initialize();
    }
    
    pthread_create(&py_thread_state.thread_id, NULL, python_thread_func, NULL);
}

void shutdown_python_thread(void) {
    // 1. Signal thread to stop
    py_thread_state.running = false;
    
    // 2. Wake up any waiting threads
    pthread_mutex_lock(&py_thread_state.queue.mutex);
    pthread_cond_broadcast(&py_thread_state.queue.not_empty);
    pthread_cond_broadcast(&py_thread_state.queue.not_full);
    pthread_mutex_unlock(&py_thread_state.queue.mutex);
    
    
    // 4. Join the thread and clean up resources
    pthread_join(py_thread_state.thread_id, NULL);
    destroy_request_queue(&py_thread_state.queue);
}

int submit_frame_feedback_request(int picture_number, int temporal_layer_index, int qp, int avg_qp,
                                 double luma_psnr, double cb_psnr, double cr_psnr,
                                 double mse_y, double mse_u, double mse_v,
                                 double luma_ssim, double cb_ssim, double cr_ssim,
                                 int picture_stream_size) {
    
    PyRequest* request = enqueue_request(&py_thread_state.queue);
    if (!request) return -1;
    
    request->type = PY_REQUEST_FRAME_FEEDBACK;
    request->params.frame_feedback.picture_number = picture_number;
    request->params.frame_feedback.temporal_layer_index = temporal_layer_index;
    request->params.frame_feedback.qp = qp;
    request->params.frame_feedback.avg_qp = avg_qp;
    request->params.frame_feedback.luma_psnr = luma_psnr;
    request->params.frame_feedback.cb_psnr = cb_psnr;
    request->params.frame_feedback.cr_psnr = cr_psnr;
    request->params.frame_feedback.mse_y = mse_y;
    request->params.frame_feedback.mse_u = mse_u;
    request->params.frame_feedback.mse_v = mse_v;
    request->params.frame_feedback.luma_ssim = luma_ssim;
    request->params.frame_feedback.cb_ssim = cb_ssim;
    request->params.frame_feedback.cr_ssim = cr_ssim;
    request->params.frame_feedback.picture_stream_size = picture_stream_size;
    
    pthread_mutex_lock(&request->mutex);
    while (!request->completed && py_thread_state.running) {
        pthread_cond_wait(&request->cond, &request->mutex);
    }
    pthread_mutex_unlock(&request->mutex);

    pthread_mutex_destroy(&request->mutex);
    pthread_cond_destroy(&request->cond);
    
    return 0;
}

int submit_sb_feedback_request(int picture_number, int sb_index, unsigned sb_origin_x, unsigned sb_origin_y,
    double luma_psnr, double cb_psnr, double cr_psnr,
    double mse_y, double mse_u, double mse_v,
    double luma_ssim, double cb_ssim, double cr_ssim,
    uint8_t* buffer_y, uint8_t* buffer_cb, uint8_t* buffer_cr,
    uint16_t sb_width, uint16_t sb_height) {

    PyRequest* request = enqueue_request(&py_thread_state.queue);
    if (!request) return -1;

    request->type = PY_REQUEST_SB_FEEDBACK;
    request->params.sb_feedback.picture_number = picture_number;
    request->params.sb_feedback.sb_index = sb_index;
    request->params.sb_feedback.sb_origin_x = sb_origin_x;
    request->params.sb_feedback.sb_origin_y = sb_origin_y;
    request->params.sb_feedback.luma_psnr = luma_psnr;
    request->params.sb_feedback.cb_psnr = cb_psnr;
    request->params.sb_feedback.cr_psnr = cr_psnr;
    request->params.sb_feedback.mse_y = mse_y;
    request->params.sb_feedback.mse_u = mse_u;
    request->params.sb_feedback.mse_v = mse_v;
    request->params.sb_feedback.luma_ssim = luma_ssim;
    request->params.sb_feedback.cb_ssim = cb_ssim;
    request->params.sb_feedback.cr_ssim = cr_ssim;
    request->params.sb_feedback.buffer_y = buffer_y;
    request->params.sb_feedback.buffer_cb = buffer_cb;
    request->params.sb_feedback.buffer_cr = buffer_cr;
    request->params.sb_feedback.sb_width = sb_width;
    request->params.sb_feedback.sb_height = sb_height;

    pthread_mutex_lock(&request->mutex);
    while (!request->completed && py_thread_state.running) {
        pthread_cond_wait(&request->cond, &request->mutex);
    }
    pthread_mutex_unlock(&request->mutex);

    pthread_mutex_destroy(&request->mutex);
    pthread_cond_destroy(&request->cond);

    return 0;
}

int submit_sb_offset_request(unsigned sb_index, unsigned sb_origin_x, unsigned sb_origin_y,
                            int sb_qp, int sb_final_blk_cnt, int mi_row_start, int mi_row_end,
                            int mi_col_start, int mi_col_end, int tg_horz_boundary,
                            int tile_row, int tile_col, int tile_rs_index,
                            int picture_number,
                            uint8_t* buffer_y, uint8_t* buffer_cb, uint8_t* buffer_cr,
                            uint16_t sb_width, uint16_t sb_height,
                            int encoder_bit_depth, int qindex, double beta, int type) {
    
    PyRequest* request = enqueue_request(&py_thread_state.queue);
    if (!request) return -1;
    
    request->type = PY_REQUEST_SB_OFFSET;
    request->params.sb_offset.sb_index = sb_index;
    request->params.sb_offset.sb_origin_x = sb_origin_x;
    request->params.sb_offset.sb_origin_y = sb_origin_y;
    request->params.sb_offset.sb_qp = sb_qp;
    request->params.sb_offset.sb_final_blk_cnt = sb_final_blk_cnt;
    request->params.sb_offset.mi_row_start = mi_row_start;
    request->params.sb_offset.mi_row_end = mi_row_end;
    request->params.sb_offset.mi_col_start = mi_col_start;
    request->params.sb_offset.mi_col_end = mi_col_end;
    request->params.sb_offset.tg_horz_boundary = tg_horz_boundary;
    request->params.sb_offset.tile_row = tile_row;
    request->params.sb_offset.tile_col = tile_col;
    request->params.sb_offset.tile_rs_index = tile_rs_index;
    request->params.sb_offset.buffer_y = buffer_y;
    request->params.sb_offset.buffer_cb = buffer_cb;
    request->params.sb_offset.buffer_cr = buffer_cr;
    request->params.sb_offset.sb_width = sb_width;
    request->params.sb_offset.sb_height = sb_height;
    request->params.sb_offset.picture_number = picture_number;
    request->params.sb_offset.encoder_bit_depth = encoder_bit_depth;
    request->params.sb_offset.qindex = qindex;
    request->params.sb_offset.beta = beta;
    request->params.sb_offset.type = type;
    
    pthread_mutex_lock(&request->mutex);
    while (!request->completed && py_thread_state.running) {
        pthread_cond_wait(&request->cond, &request->mutex);
    }
    pthread_mutex_unlock(&request->mutex);
    
    int result = request->result.int_result;
    
    pthread_mutex_destroy(&request->mutex);
    pthread_cond_destroy(&request->cond);
    
    return result;
}


void cleanup_python_thread_objects(void) {
    py_thread_state.frame_feedback_func = NULL;
    py_thread_state.sb_feedback_func = NULL;
    py_thread_state.sb_offset_func = NULL;
}

void set_python_thread_running(bool running) {
    py_thread_state.running = running;
}

void signal_python_thread_termination(void) {
    // Signal the thread to terminate
    pthread_mutex_lock(&py_thread_state.queue.mutex);
    pthread_cond_broadcast(&py_thread_state.queue.not_empty);
    pthread_cond_broadcast(&py_thread_state.queue.not_full);
    pthread_mutex_unlock(&py_thread_state.queue.mutex);
}

bool is_python_thread_running(void) {
    return py_thread_state.running;
}