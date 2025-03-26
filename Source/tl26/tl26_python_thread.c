#include "tl26_python_thread.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// 全局线程状态
TL26ThreadState py_thread_state = {0};

// 请求ID计数器
static atomic_int global_request_counter = 0;

// 请求结构体定义
// SB反馈请求结构体
typedef struct {
    int request_id;
    int picture_number;
    unsigned sb_index;
    unsigned sb_origin_x;
    unsigned sb_origin_y;
    uint16_t sb_width;
    uint16_t sb_height;
    
    // 指标数据
    double luma_psnr;
    double cb_psnr;
    double cr_psnr;
    double mse_y;
    double mse_u;
    double mse_v;
    double luma_ssim;
    double cb_ssim;
    double cr_ssim;
    
    // 缓冲区的完整拷贝
    uint8_t* y_buffer;
    uint8_t* cb_buffer;
    uint8_t* cr_buffer;
    size_t y_buffer_size;
    size_t cb_cr_buffer_size;
    
    // 完成标志
    volatile bool completed;
    volatile bool success;
    
    // 同步对象
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} SbFeedbackRequest;

// SB偏移请求结构体
typedef struct {
    int request_id;
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
    
    // 缓冲区
    uint8_t* y_buffer;
    uint8_t* cb_buffer;
    uint8_t* cr_buffer;
    uint16_t sb_width;
    uint16_t sb_height;
    size_t y_buffer_size;
    size_t cb_cr_buffer_size;
    
    int picture_number;
    int encoder_bit_depth;
    int qindex;
    double beta;
    int type;
    
    // 结果
    volatile bool completed;
    int result;
    
    // 同步对象
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} SbOffsetRequest;

// 帧反馈请求结构体
typedef struct {
    int request_id;
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
    
    // 完成标志
    volatile bool completed;
    volatile bool success;
    
    // 同步对象
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} FrameFeedbackRequest;

// 安全的Python列表创建函数
PyObject* create_python_list_from_buffer(uint8_t* buffer, int width, int height) {
    if (!buffer || width <= 0 || height <= 0) {
        PyErr_SetString(PyExc_ValueError, "Invalid buffer or dimensions");
        return NULL;
    }
    
    PyObject* list = PyList_New(height);
    if (!list) return NULL;
    
    for (int i = 0; i < height; i++) {
        PyObject* row = PyList_New(width);
        if (!row) {
            Py_DECREF(list);
            return NULL;
        }
        
        for (int j = 0; j < width; j++) {
            // 计算索引
            int index = i * width + j;
            PyObject* value = PyLong_FromLong((long)buffer[index]);
            if (!value) {
                Py_DECREF(row);
                Py_DECREF(list);
                return NULL;
            }
            
            PyList_SetItem(row, j, value);
        }
        
        PyList_SetItem(list, i, row);
    }
    
    return list;
}

// SB反馈工作线程函数
static void* sb_feedback_thread_func(void* arg) {
    SbFeedbackRequest* req = (SbFeedbackRequest*)arg;
    
    fprintf(stderr, "[Worker-%d] Starting processing\n", req->request_id);
    
    // 确保请求有效
    if (!req) {
        fprintf(stderr, "[Worker] Error: NULL request\n");
        return NULL;
    }
    
    // 验证请求参数
    if (!req->y_buffer || !req->cb_buffer || !req->cr_buffer || 
        req->sb_width <= 0 || req->sb_height <= 0) {
        fprintf(stderr, "[Worker-%d] Invalid buffers in request\n", req->request_id);
        req->success = false;
        goto cleanup;
    }
    
    // 获取GIL - 使用PyGILState API
    fprintf(stderr, "[Worker-%d] Acquiring GIL...\n", req->request_id);
    PyGILState_STATE gstate = PyGILState_Ensure();
    fprintf(stderr, "[Worker-%d] GIL acquired\n", req->request_id);
    
    // 检查Python状态
    if (!Py_IsInitialized()) {
        fprintf(stderr, "[Worker-%d] Error: Python is not initialized\n", req->request_id);
        PyGILState_Release(gstate);
        req->success = false;
        goto cleanup;
    }
    
    // 检查回调函数
    if (!py_thread_state.sb_feedback_func) {
        fprintf(stderr, "[Worker-%d] Error: sb_feedback_func is NULL\n", req->request_id);
        PyGILState_Release(gstate);
        req->success = false;
        goto cleanup;
    }
    
    // 创建Python列表
    PyObject* py_y = NULL;
    PyObject* py_cb = NULL;
    PyObject* py_cr = NULL;
    
    fprintf(stderr, "[Worker-%d] Creating Python buffer lists\n", req->request_id);
    py_y = create_python_list_from_buffer(req->y_buffer, req->sb_width, req->sb_height);
    if (!py_y) {
        fprintf(stderr, "[Worker-%d] Failed to create Y buffer list\n", req->request_id);
        PyGILState_Release(gstate);
        req->success = false;
        goto cleanup;
    }
    
    py_cb = create_python_list_from_buffer(req->cb_buffer, req->sb_width/2, req->sb_height/2);
    if (!py_cb) {
        fprintf(stderr, "[Worker-%d] Failed to create Cb buffer list\n", req->request_id);
        Py_DECREF(py_y);
        PyGILState_Release(gstate);
        req->success = false;
        goto cleanup;
    }
    
    py_cr = create_python_list_from_buffer(req->cr_buffer, req->sb_width/2, req->sb_height/2);
    if (!py_cr) {
        fprintf(stderr, "[Worker-%d] Failed to create Cr buffer list\n", req->request_id);
        Py_DECREF(py_y);
        Py_DECREF(py_cb);
        PyGILState_Release(gstate);
        req->success = false;
        goto cleanup;
    }
    
    fprintf(stderr, "[Worker-%d] Building Python function arguments\n", req->request_id);
    
    // 调用Python函数
    PyObject* args = Py_BuildValue(
        "iIIIdddddddddOOO",
        req->picture_number,
        req->sb_index,
        req->sb_origin_x,
        req->sb_origin_y,
        req->luma_psnr,
        req->cb_psnr,
        req->cr_psnr,
        req->mse_y,
        req->mse_u,
        req->mse_v,
        req->luma_ssim,
        req->cb_ssim,
        req->cr_ssim,
        py_y, py_cb, py_cr);
    
    if (!args) {
        fprintf(stderr, "[Worker-%d] Failed to build arguments\n", req->request_id);
        Py_DECREF(py_y);
        Py_DECREF(py_cb);
        Py_DECREF(py_cr);
        PyGILState_Release(gstate);
        req->success = false;
        goto cleanup;
    }
    
    fprintf(stderr, "[Worker-%d] Calling Python function\n", req->request_id);
    PyObject* result = PyObject_CallObject(py_thread_state.sb_feedback_func, args);
    
    if (result) {
        fprintf(stderr, "[Worker-%d] Python function call succeeded\n", req->request_id);
        Py_DECREF(result);
        req->success = true;
    } else {
        fprintf(stderr, "[Worker-%d] Python function call failed\n", req->request_id);
        PyErr_Print();
        req->success = false;
    }
    
    // 清理Python对象
    Py_DECREF(args);
    Py_DECREF(py_y);
    Py_DECREF(py_cb);
    Py_DECREF(py_cr);
    
    fprintf(stderr, "[Worker-%d] Releasing GIL\n", req->request_id);
    PyGILState_Release(gstate);
    fprintf(stderr, "[Worker-%d] GIL released\n", req->request_id);
    
cleanup:
    // 标记请求完成并通知等待线程
    pthread_mutex_lock(&req->mutex);
    req->completed = true;
    pthread_cond_signal(&req->cond);
    pthread_mutex_unlock(&req->mutex);
    
    fprintf(stderr, "[Worker-%d] Request %s\n", req->request_id, 
            req->success ? "succeeded" : "failed");
    
    return NULL;
}

// SB偏移工作线程函数
static void* sb_offset_thread_func(void* arg) {
    SbOffsetRequest* req = (SbOffsetRequest*)arg;
    
    fprintf(stderr, "[Worker-%d] Starting offset processing\n", req->request_id);
    
    // 确保请求有效
    if (!req) {
        fprintf(stderr, "[Worker] Error: NULL offset request\n");
        return NULL;
    }
    
    // 验证请求参数
    if (!req->y_buffer || !req->cb_buffer || !req->cr_buffer || 
        req->sb_width <= 0 || req->sb_height <= 0) {
        fprintf(stderr, "[Worker-%d] Invalid buffers in offset request\n", req->request_id);
        req->result = -1;
        goto cleanup;
    }
    
    // 获取GIL - 使用PyGILState API
    fprintf(stderr, "[Worker-%d] Acquiring GIL for offset request...\n", req->request_id);
    PyGILState_STATE gstate = PyGILState_Ensure();
    fprintf(stderr, "[Worker-%d] GIL acquired for offset request\n", req->request_id);
    
    // 检查Python状态
    if (!Py_IsInitialized()) {
        fprintf(stderr, "[Worker-%d] Error: Python is not initialized\n", req->request_id);
        PyGILState_Release(gstate);
        req->result = -1;
        goto cleanup;
    }
    
    // 检查回调函数
    if (!py_thread_state.sb_offset_func) {
        fprintf(stderr, "[Worker-%d] Error: sb_offset_func is NULL\n", req->request_id);
        PyGILState_Release(gstate);
        req->result = -1;
        goto cleanup;
    }
    
    // 创建Python列表
    PyObject* py_y = NULL;
    PyObject* py_cb = NULL;
    PyObject* py_cr = NULL;
    
    fprintf(stderr, "[Worker-%d] Creating Python buffer lists for offset\n", req->request_id);
    py_y = create_python_list_from_buffer(req->y_buffer, req->sb_width, req->sb_height);
    if (!py_y) {
        fprintf(stderr, "[Worker-%d] Failed to create Y buffer list for offset\n", req->request_id);
        PyGILState_Release(gstate);
        req->result = -1;
        goto cleanup;
    }
    
    py_cb = create_python_list_from_buffer(req->cb_buffer, req->sb_width/2, req->sb_height/2);
    if (!py_cb) {
        fprintf(stderr, "[Worker-%d] Failed to create Cb buffer list for offset\n", req->request_id);
        Py_DECREF(py_y);
        PyGILState_Release(gstate);
        req->result = -1;
        goto cleanup;
    }
    
    py_cr = create_python_list_from_buffer(req->cr_buffer, req->sb_width/2, req->sb_height/2);
    if (!py_cr) {
        fprintf(stderr, "[Worker-%d] Failed to create Cr buffer list for offset\n", req->request_id);
        Py_DECREF(py_y);
        Py_DECREF(py_cb);
        PyGILState_Release(gstate);
        req->result = -1;
        goto cleanup;
    }
    
    fprintf(stderr, "[Worker-%d] Building Python function arguments for offset\n", req->request_id);
    
    // 调用Python函数
    PyObject* args = Py_BuildValue(
        "IIIiiiiiiiiiiiiidiOOO",
        req->sb_index,
        req->sb_origin_x,
        req->sb_origin_y,
        req->sb_qp,
        req->sb_final_blk_cnt,
        req->mi_row_start,
        req->mi_row_end,
        req->mi_col_start,
        req->mi_col_end,
        req->tg_horz_boundary,
        req->tile_row,
        req->tile_col,
        req->tile_rs_index,
        req->picture_number,
        req->encoder_bit_depth,
        req->qindex,
        req->beta,
        req->type,
        py_y, py_cb, py_cr);
    
    if (!args) {
        fprintf(stderr, "[Worker-%d] Failed to build arguments for offset\n", req->request_id);
        Py_DECREF(py_y);
        Py_DECREF(py_cb);
        Py_DECREF(py_cr);
        PyGILState_Release(gstate);
        req->result = -1;
        goto cleanup;
    }
    
    fprintf(stderr, "[Worker-%d] Calling Python offset function\n", req->request_id);
    PyObject* result = PyObject_CallObject(py_thread_state.sb_offset_func, args);
    
    if (result) {
        req->result = PyLong_AsLong(result);
        fprintf(stderr, "[Worker-%d] Python offset function call succeeded, result: %d\n", 
                req->request_id, req->result);
        Py_DECREF(result);
    } else {
        fprintf(stderr, "[Worker-%d] Python offset function call failed\n", req->request_id);
        PyErr_Print();
        req->result = -1;
    }
    
    // 清理Python对象
    Py_DECREF(args);
    Py_DECREF(py_y);
    Py_DECREF(py_cb);
    Py_DECREF(py_cr);
    
    fprintf(stderr, "[Worker-%d] Releasing GIL for offset\n", req->request_id);
    PyGILState_Release(gstate);
    fprintf(stderr, "[Worker-%d] GIL released for offset\n", req->request_id);
    
cleanup:
    // 标记请求完成并通知等待线程
    pthread_mutex_lock(&req->mutex);
    req->completed = true;
    pthread_cond_signal(&req->cond);
    pthread_mutex_unlock(&req->mutex);
    
    fprintf(stderr, "[Worker-%d] Offset request completed with result %d\n", 
            req->request_id, req->result);
    
    return NULL;
}

// 帧反馈工作线程函数
static void* frame_feedback_thread_func(void* arg) {
    FrameFeedbackRequest* req = (FrameFeedbackRequest*)arg;
    
    fprintf(stderr, "[Worker-%d] Starting frame feedback processing\n", req->request_id);
    
    // 确保请求有效
    if (!req) {
        fprintf(stderr, "[Worker] Error: NULL frame request\n");
        return NULL;
    }
    
    // 获取GIL - 使用PyGILState API
    fprintf(stderr, "[Worker-%d] Acquiring GIL for frame feedback...\n", req->request_id);
    PyGILState_STATE gstate = PyGILState_Ensure();
    fprintf(stderr, "[Worker-%d] GIL acquired for frame feedback\n", req->request_id);
    
    // 检查Python状态
    if (!Py_IsInitialized()) {
        fprintf(stderr, "[Worker-%d] Error: Python is not initialized\n", req->request_id);
        PyGILState_Release(gstate);
        req->success = false;
        goto cleanup;
    }
    
    // 检查回调函数
    if (!py_thread_state.frame_feedback_func) {
        fprintf(stderr, "[Worker-%d] Error: frame_feedback_func is NULL\n", req->request_id);
        PyGILState_Release(gstate);
        req->success = false;
        goto cleanup;
    }
    
    fprintf(stderr, "[Worker-%d] Building Python function arguments for frame\n", req->request_id);
    
    // 调用Python函数
    PyObject* args = Py_BuildValue(
        "iiiidddddddddi",
        req->picture_number,
        req->temporal_layer_index,
        req->qp,
        req->avg_qp,
        req->luma_psnr,
        req->cb_psnr,
        req->cr_psnr,
        req->mse_y,
        req->mse_u,
        req->mse_v,
        req->luma_ssim,
        req->cb_ssim,
        req->cr_ssim,
        req->picture_stream_size);
    
    if (!args) {
        fprintf(stderr, "[Worker-%d] Failed to build arguments for frame\n", req->request_id);
        PyGILState_Release(gstate);
        req->success = false;
        goto cleanup;
    }
    
    fprintf(stderr, "[Worker-%d] Calling Python frame function\n", req->request_id);
    PyObject* result = PyObject_CallObject(py_thread_state.frame_feedback_func, args);
    
    if (result) {
        fprintf(stderr, "[Worker-%d] Python frame function call succeeded\n", req->request_id);
        Py_DECREF(result);
        req->success = true;
    } else {
        fprintf(stderr, "[Worker-%d] Python frame function call failed\n", req->request_id);
        PyErr_Print();
        req->success = false;
    }
    
    // 清理Python对象
    Py_DECREF(args);
    
    fprintf(stderr, "[Worker-%d] Releasing GIL for frame\n", req->request_id);
    PyGILState_Release(gstate);
    fprintf(stderr, "[Worker-%d] GIL released for frame\n", req->request_id);
    
cleanup:
    // 标记请求完成并通知等待线程
    pthread_mutex_lock(&req->mutex);
    req->completed = true;
    pthread_cond_signal(&req->cond);
    pthread_mutex_unlock(&req->mutex);
    
    fprintf(stderr, "[Worker-%d] Frame request %s\n", req->request_id, 
            req->success ? "succeeded" : "failed");
    
    return NULL;
}

// 初始化Python线程
void init_python_thread(void) {
    fprintf(stderr, "Initializing Python environment...\n");
    
    // 1. 确保Python已正确初始化
    if (!Py_IsInitialized()) {
        fprintf(stderr, "Python not initialized, calling Py_Initialize()\n");
        Py_Initialize();
    }
    
    // 2. 使用PyGILState API确保我们持有GIL
    fprintf(stderr, "Ensuring GIL is held...\n");
    PyGILState_STATE gstate = PyGILState_Ensure();
    fprintf(stderr, "GIL acquired successfully\n");
    
    // 3. 设置Python模块搜索路径
    fprintf(stderr, "Setting up module search paths...\n");
    PyObject* sysPath = PySys_GetObject("path");  // 借用引用，不要减少引用计数
    if (sysPath != NULL) {
        // 添加当前目录
        PyObject* currPath = PyUnicode_FromString(".");
        PyList_Append(sysPath, currPath);
        Py_DECREF(currPath);
        
        // 如果有特定的模块目录，也添加它
        const char* modulePath = getenv("TL26_MODULE_PATH");
        if (modulePath != NULL) {
            PyObject* modPath = PyUnicode_FromString(modulePath);
            PyList_Append(sysPath, modPath);
            Py_DECREF(modPath);
            fprintf(stderr, "Added module path from TL26_MODULE_PATH: %s\n", modulePath);
        }
        
        // 打印当前Python路径
        fprintf(stderr, "Current Python path:\n");
        Py_ssize_t pathSize = PyList_Size(sysPath);
        for (Py_ssize_t i = 0; i < pathSize; i++) {
            PyObject* pathItem = PyList_GetItem(sysPath, i);  // 借用引用
            const char* pathStr = PyUnicode_AsUTF8(pathItem);
            fprintf(stderr, "  %s\n", pathStr);
        }
    }
    
    // 4. 导入模块，加载函数
    fprintf(stderr, "Trying to import modules...\n");
    py_thread_state.frame_feedback_func = NULL;
    py_thread_state.sb_feedback_func = NULL;
    py_thread_state.sb_offset_func = NULL;
    
    PyObject* feedback_module = PyImport_ImportModule("tl26.feedback");
    if (feedback_module) {
        fprintf(stderr, "Successfully imported tl26.feedback module\n");
        py_thread_state.frame_feedback_func = PyObject_GetAttrString(feedback_module, "frame_report_feedback");
        py_thread_state.sb_feedback_func = PyObject_GetAttrString(feedback_module, "sb_report_feedback");
        Py_DECREF(feedback_module);
    } else {
        fprintf(stderr, "Failed to import tl26.feedback module, error:\n");
        PyErr_Print();
    }
    
    PyObject* request_module = PyImport_ImportModule("tl26.request");
    if (request_module) {
        fprintf(stderr, "Successfully imported tl26.request module\n");
        py_thread_state.sb_offset_func = PyObject_GetAttrString(request_module, "sb_send_offset_request");
        Py_DECREF(request_module);
    } else {
        fprintf(stderr, "Failed to import tl26.request module, error:\n");
        PyErr_Print();
    }
    
    // 检查导入的函数是否有效
    if (py_thread_state.frame_feedback_func && !PyCallable_Check(py_thread_state.frame_feedback_func)) {
        fprintf(stderr, "Warning: frame_report_feedback is not callable\n");
        Py_DECREF(py_thread_state.frame_feedback_func);
        py_thread_state.frame_feedback_func = NULL;
    }
    
    if (py_thread_state.sb_feedback_func && !PyCallable_Check(py_thread_state.sb_feedback_func)) {
        fprintf(stderr, "Warning: sb_report_feedback is not callable\n");
        Py_DECREF(py_thread_state.sb_feedback_func);
        py_thread_state.sb_feedback_func = NULL;
    }
    
    if (py_thread_state.sb_offset_func && !PyCallable_Check(py_thread_state.sb_offset_func)) {
        fprintf(stderr, "Warning: sb_send_offset_request is not callable\n");
        Py_DECREF(py_thread_state.sb_offset_func);
        py_thread_state.sb_offset_func = NULL;
    }
    
    // 5. 初始化请求计数器
    atomic_store(&global_request_counter, 0);
    
    // 6. 重要！释放GIL
    fprintf(stderr, "Releasing GIL\n");
    PyGILState_Release(gstate);
    fprintf(stderr, "GIL released successfully\n");
    
    // 7. 设置运行标志
    py_thread_state.running = true;
    
    fprintf(stderr, "Python thread system initialized successfully\n");
}

// 提交SB反馈请求
int submit_sb_feedback_request(int picture_number, int sb_index, unsigned sb_origin_x, unsigned sb_origin_y,
    double luma_psnr, double cb_psnr, double cr_psnr,
    double mse_y, double mse_u, double mse_v,
    double luma_ssim, double cb_ssim, double cr_ssim,
    uint8_t* buffer_y, uint8_t* buffer_cb, uint8_t* buffer_cr,
    uint16_t sb_width, uint16_t sb_height) {
    
    // 检查Python函数是否可用
    if (!py_thread_state.sb_feedback_func) {
        fprintf(stderr, "[Main] Python sb_feedback_func not available, skipping request\n");
        return 0; // 返回成功以允许程序继续运行
    }
    
    int request_id = atomic_fetch_add(&global_request_counter, 1);
    
    fprintf(stderr, "[Main] Creating request %d for SB [%d,%d] size %dx%d\n", 
            request_id, sb_origin_x, sb_origin_y, sb_width, sb_height);
    
    // 参数验证
    if (!buffer_y || !buffer_cb || !buffer_cr || sb_width <= 0 || sb_height <= 0) {
        fprintf(stderr, "[Main] Invalid parameters in submit_sb_feedback_request\n");
        return -1;
    }
    
    // 创建并初始化请求对象
    SbFeedbackRequest* req = (SbFeedbackRequest*)malloc(sizeof(SbFeedbackRequest));
    if (!req) {
        fprintf(stderr, "[Main] Failed to allocate request memory\n");
        return -1;
    }
    
    // 初始化同步对象
    pthread_mutex_init(&req->mutex, NULL);
    pthread_cond_init(&req->cond, NULL);
    
    // 设置请求属性
    req->request_id = request_id;
    req->picture_number = picture_number;
    req->sb_index = sb_index;
    req->sb_origin_x = sb_origin_x;
    req->sb_origin_y = sb_origin_y;
    req->sb_width = sb_width;
    req->sb_height = sb_height;
    req->luma_psnr = luma_psnr;
    req->cb_psnr = cb_psnr;
    req->cr_psnr = cr_psnr;
    req->mse_y = mse_y;
    req->mse_u = mse_u;
    req->mse_v = mse_v;
    req->luma_ssim = luma_ssim;
    req->cb_ssim = cb_ssim;
    req->cr_ssim = cr_ssim;
    req->completed = false;
    req->success = false;
    
    // 分配并复制缓冲区
    size_t y_size = sb_width * sb_height;
    size_t cb_cr_size = (sb_width/2) * (sb_height/2);
    
    req->y_buffer = (uint8_t*)malloc(y_size);
    req->cb_buffer = (uint8_t*)malloc(cb_cr_size);
    req->cr_buffer = (uint8_t*)malloc(cb_cr_size);
    
    if (!req->y_buffer || !req->cb_buffer || !req->cr_buffer) {
        fprintf(stderr, "[Main] Failed to allocate buffer memory for request %d\n", request_id);
        if (req->y_buffer) free(req->y_buffer);
        if (req->cb_buffer) free(req->cb_buffer);
        if (req->cr_buffer) free(req->cr_buffer);
        pthread_mutex_destroy(&req->mutex);
        pthread_cond_destroy(&req->cond);
        free(req);
        return -1;
    }
    
    // 复制缓冲区数据
    memcpy(req->y_buffer, buffer_y, y_size);
    memcpy(req->cb_buffer, buffer_cb, cb_cr_size);
    memcpy(req->cr_buffer, buffer_cr, cb_cr_size);
    
    req->y_buffer_size = y_size;
    req->cb_cr_buffer_size = cb_cr_size;
    
    // 创建专用工作线程处理此请求
    pthread_t worker_thread;
    int thread_result = pthread_create(&worker_thread, NULL, sb_feedback_thread_func, req);
    if (thread_result != 0) {
        fprintf(stderr, "[Main] Failed to create worker thread for request %d: %s\n", 
                request_id, strerror(thread_result));
        free(req->y_buffer);
        free(req->cb_buffer);
        free(req->cr_buffer);
        pthread_mutex_destroy(&req->mutex);
        pthread_cond_destroy(&req->cond);
        free(req);
        return -1;
    }
    
    // 分离线程，让它自行退出
    pthread_detach(worker_thread);
    
    // 等待请求完成
    pthread_mutex_lock(&req->mutex);
    while (!req->completed && py_thread_state.running) {
        pthread_cond_wait(&req->cond, &req->mutex);
    }
    pthread_mutex_unlock(&req->mutex);
    
    // 获取结果
    int result = req->success ? 0 : -1;
    
    // 清理资源
    free(req->y_buffer);
    free(req->cb_buffer);
    free(req->cr_buffer);
    pthread_mutex_destroy(&req->mutex);
    pthread_cond_destroy(&req->cond);
    free(req);
    
    fprintf(stderr, "[Main] Request %d finished with result %d\n", request_id, result);
    return result;
}

// 提交SB偏移请求
int submit_sb_offset_request(unsigned sb_index, unsigned sb_origin_x, unsigned sb_origin_y,
                            int sb_qp, int sb_final_blk_cnt, int mi_row_start, int mi_row_end,
                            int mi_col_start, int mi_col_end, int tg_horz_boundary,
                            int tile_row, int tile_col, int tile_rs_index,
                            int picture_number,
                            uint8_t* buffer_y, uint8_t* buffer_cb, uint8_t* buffer_cr,
                            uint16_t sb_width, uint16_t sb_height,
                            int encoder_bit_depth, int qindex, double beta, int type) {
    
    // 检查Python函数是否可用
    if (!py_thread_state.sb_offset_func) {
        fprintf(stderr, "[Main] Python sb_offset_func not available, skipping request\n");
        return 0; // 返回值由实际需求决定
    }
    
    int request_id = atomic_fetch_add(&global_request_counter, 1);
    
    fprintf(stderr, "[Main] Creating offset request %d for SB [%d,%d] size %dx%d\n", 
            request_id, sb_origin_x, sb_origin_y, sb_width, sb_height);
    
    // 参数验证
    if (!buffer_y || !buffer_cb || !buffer_cr || sb_width <= 0 || sb_height <= 0) {
        fprintf(stderr, "[Main] Invalid parameters in submit_sb_offset_request\n");
        return -1;
    }
    
    // 创建并初始化请求对象
    SbOffsetRequest* req = (SbOffsetRequest*)malloc(sizeof(SbOffsetRequest));
    if (!req) {
        fprintf(stderr, "[Main] Failed to allocate offset request memory\n");
        return -1;
    }
    
    // 初始化同步对象
    pthread_mutex_init(&req->mutex, NULL);
    pthread_cond_init(&req->cond, NULL);
    
    // 设置请求属性
    req->request_id = request_id;
    req->sb_index = sb_index;
    req->sb_origin_x = sb_origin_x;
    req->sb_origin_y = sb_origin_y;
    req->sb_qp = sb_qp;
    req->sb_final_blk_cnt = sb_final_blk_cnt;
    req->mi_row_start = mi_row_start;
    req->mi_row_end = mi_row_end;
    req->mi_col_start = mi_col_start;
    req->mi_col_end = mi_col_end;
    req->tg_horz_boundary = tg_horz_boundary;
    req->tile_row = tile_row;
    req->tile_col = tile_col;
    req->tile_rs_index = tile_rs_index;
    req->picture_number = picture_number;
    req->encoder_bit_depth = encoder_bit_depth;
    req->qindex = qindex;
    req->beta = beta;
    req->type = type;
    req->sb_width = sb_width;
    req->sb_height = sb_height;
    req->completed = false;
    req->result = -1;
    
    // 分配并复制缓冲区
    size_t y_size = sb_width * sb_height;
    size_t cb_cr_size = (sb_width/2) * (sb_height/2);
    
    req->y_buffer = (uint8_t*)malloc(y_size);
    req->cb_buffer = (uint8_t*)malloc(cb_cr_size);
    req->cr_buffer = (uint8_t*)malloc(cb_cr_size);
    
    if (!req->y_buffer || !req->cb_buffer || !req->cr_buffer) {
        fprintf(stderr, "[Main] Failed to allocate buffer memory for offset request %d\n", request_id);
        if (req->y_buffer) free(req->y_buffer);
        if (req->cb_buffer) free(req->cb_buffer);
        if (req->cr_buffer) free(req->cr_buffer);
        pthread_mutex_destroy(&req->mutex);
        pthread_cond_destroy(&req->cond);
        free(req);
        return -1;
    }
    
    // 复制缓冲区数据
    memcpy(req->y_buffer, buffer_y, y_size);
    memcpy(req->cb_buffer, buffer_cb, cb_cr_size);
    memcpy(req->cr_buffer, buffer_cr, cb_cr_size);
    
    req->y_buffer_size = y_size;
    req->cb_cr_buffer_size = cb_cr_size;
    
    // 创建专用工作线程处理此请求
    pthread_t worker_thread;
    int thread_result = pthread_create(&worker_thread, NULL, sb_offset_thread_func, req);
    if (thread_result != 0) {
        fprintf(stderr, "[Main] Failed to create worker thread for offset request %d: %s\n", 
                request_id, strerror(thread_result));
        free(req->y_buffer);
        free(req->cb_buffer);
        free(req->cr_buffer);
        pthread_mutex_destroy(&req->mutex);
        pthread_cond_destroy(&req->cond);
        free(req);
        return -1;
    }
    
    // 分离线程，让它自行退出
    pthread_detach(worker_thread);
    
    // 等待请求完成
    pthread_mutex_lock(&req->mutex);
    while (!req->completed && py_thread_state.running) {
        pthread_cond_wait(&req->cond, &req->mutex);
    }
    pthread_mutex_unlock(&req->mutex);
    
    // 获取结果
    int result = req->result;
    
    // 清理资源
    free(req->y_buffer);
    free(req->cb_buffer);
    free(req->cr_buffer);
    pthread_mutex_destroy(&req->mutex);
    pthread_cond_destroy(&req->cond);
    free(req);
    
    fprintf(stderr, "[Main] Offset request %d finished with result %d\n", request_id, result);
    return result;
}

// 提交帧反馈请求
int submit_frame_feedback_request(int picture_number, int temporal_layer_index, int qp, int avg_qp,
                                 double luma_psnr, double cb_psnr, double cr_psnr,
                                 double mse_y, double mse_u, double mse_v,
                                 double luma_ssim, double cb_ssim, double cr_ssim,
                                 int picture_stream_size) {
    
    // 检查Python函数是否可用
    if (!py_thread_state.frame_feedback_func) {
        fprintf(stderr, "[Main] Python frame_feedback_func not available, skipping request\n");
        return 0; // 返回成功以允许程序继续运行
    }
    
    int request_id = atomic_fetch_add(&global_request_counter, 1);
    
    fprintf(stderr, "[Main] Creating frame feedback request %d for frame %d\n", 
            request_id, picture_number);
    
    // 创建并初始化请求对象
    FrameFeedbackRequest* req = (FrameFeedbackRequest*)malloc(sizeof(FrameFeedbackRequest));
    if (!req) {
        fprintf(stderr, "[Main] Failed to allocate frame feedback request memory\n");
        return -1;
    }
    
    // 初始化同步对象
    pthread_mutex_init(&req->mutex, NULL);
    pthread_cond_init(&req->cond, NULL);
    
    // 设置请求属性
    req->request_id = request_id;
    req->picture_number = picture_number;
    req->temporal_layer_index = temporal_layer_index;
    req->qp = qp;
    req->avg_qp = avg_qp;
    req->luma_psnr = luma_psnr;
    req->cb_psnr = cb_psnr;
    req->cr_psnr = cr_psnr;
    req->mse_y = mse_y;
    req->mse_u = mse_u;
    req->mse_v = mse_v;
    req->luma_ssim = luma_ssim;
    req->cb_ssim = cb_ssim;
    req->cr_ssim = cr_ssim;
    req->picture_stream_size = picture_stream_size;
    req->completed = false;
    req->success = false;
    
    // 创建专用工作线程处理此请求
    pthread_t worker_thread;
    int thread_result = pthread_create(&worker_thread, NULL, frame_feedback_thread_func, req);
    if (thread_result != 0) {
        fprintf(stderr, "[Main] Failed to create worker thread for frame feedback request %d: %s\n", 
                request_id, strerror(thread_result));
        pthread_mutex_destroy(&req->mutex);
        pthread_cond_destroy(&req->cond);
        free(req);
        return -1;
    }
    
    // 分离线程，让它自行退出
    pthread_detach(worker_thread);
    
    // 等待请求完成
    pthread_mutex_lock(&req->mutex);
    while (!req->completed && py_thread_state.running) {
        pthread_cond_wait(&req->cond, &req->mutex);
    }
    pthread_mutex_unlock(&req->mutex);
    
    // 获取结果
    int result = req->success ? 0 : -1;
    
    // 清理资源
    pthread_mutex_destroy(&req->mutex);
    pthread_cond_destroy(&req->cond);
    free(req);
    
    fprintf(stderr, "[Main] Frame feedback request %d finished with result %d\n", request_id, result);
    return result;
}

// 关闭Python线程
void shutdown_python_thread(void) {
    fprintf(stderr, "Shutting down Python thread system\n");
    
    // 设置停止标志
    py_thread_state.running = false;
    
    // 使用PyGILState API获取GIL
    fprintf(stderr, "Acquiring GIL for shutdown\n");
    PyGILState_STATE gstate = PyGILState_Ensure();
    
    // 清理Python对象
    fprintf(stderr, "Cleaning up Python objects\n");
    Py_XDECREF(py_thread_state.frame_feedback_func);
    Py_XDECREF(py_thread_state.sb_feedback_func);
    Py_XDECREF(py_thread_state.sb_offset_func);
    
    py_thread_state.frame_feedback_func = NULL;
    py_thread_state.sb_feedback_func = NULL;
    py_thread_state.sb_offset_func = NULL;
    
    // 释放GIL
    fprintf(stderr, "Releasing GIL\n");
    PyGILState_Release(gstate);
    
    fprintf(stderr, "Python thread system shutdown complete\n");
}

// 设置线程运行标志
void set_python_thread_running(bool running) {
    py_thread_state.running = running;
}

// 获取线程运行状态
bool is_python_thread_running(void) {
    return py_thread_state.running;
}

// 通知线程终止
void signal_python_thread_termination(void) {
    py_thread_state.running = false;
}