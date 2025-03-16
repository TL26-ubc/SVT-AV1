#include "tl26_flags.h"
#include "tl26_feedback.h"

static double get_psnr_tl26(double sse, double max) {
    double psnr;
    if (sse == 0)
        psnr = 10 * log10(max / (double)0.1);
    else
        psnr = 10 * log10(max / sse);
    return psnr;
}

static void report_frame_feedback_helper(int picture_number, int temporal_layer_index, int qp, int avg_qp,
                                         double luma_psnr, double cb_psnr, double cr_psnr, double mse_y, double mse_u,
                                         double mse_v, double luma_ssim, double cb_ssim, double cr_ssim,
                                         int picture_stream_size) {
    // call the frame_report_feedback function
    if (PyCallable_Check(f_frame_report_feedback)) {
        PyObject *args = Py_BuildValue(
            "iiiidddddddddi", // this is telling the function to expect 14 arguments for int and double
            picture_number,
            temporal_layer_index,
            qp,
            avg_qp,
            luma_psnr,
            cb_psnr,
            cr_psnr,
            mse_y,
            mse_u,
            mse_v,
            luma_ssim,
            cb_ssim,
            cr_ssim,
            picture_stream_size);

        PyObject *pValue = PyObject_CallObject(f_frame_report_feedback, args);
        if (pValue != NULL) {
            Py_DECREF(pValue);
        } else {
            PyErr_Print();
        }
        Py_DECREF(args);
    } else {
        PyErr_Print();
    }
}
/**
 * Use the same calculation as how stat report function is done in @fn process_output_statistics_buffer
 * Report the python code with frame encoding results
 */
void report_frame_feedback(EbBufferHeaderType *header_ptr, EbConfig *app_cfg) {
    uint32_t max_luma_value = (app_cfg->config.encoder_bit_depth == 8) ? 255 : 1023;
#if FTR_SIGNAL_LAYER
    uint8_t temporal_layer_index;
#endif
#if FTR_SIGNAL_AVERAGE_QP
    uint32_t avg_qp;
#endif
    uint64_t picture_stream_size, luma_sse, cr_sse, cb_sse, picture_number, picture_qp;
    double   luma_ssim, cr_ssim, cb_ssim;
    double   temp_var, luma_psnr, cb_psnr, cr_psnr;
    uint32_t source_width  = app_cfg->config.source_width;
    uint32_t source_height = app_cfg->config.source_height;

    picture_stream_size = header_ptr->n_filled_len;
    luma_sse            = header_ptr->luma_sse;
    cr_sse              = header_ptr->cr_sse;
    cb_sse              = header_ptr->cb_sse;
    picture_number      = header_ptr->pts;
#if FTR_SIGNAL_LAYER
    temporal_layer_index = header_ptr->temporal_layer_index;
#endif
    picture_qp = header_ptr->qp;
#if FTR_SIGNAL_AVERAGE_QP
    avg_qp = header_ptr->avg_qp;
#endif
    luma_ssim = header_ptr->luma_ssim;
    cr_ssim   = header_ptr->cr_ssim;
    cb_ssim   = header_ptr->cb_ssim;

    temp_var = (double)max_luma_value * max_luma_value * (source_width * source_height);

    luma_psnr = get_psnr_tl26((double)luma_sse, temp_var);

    temp_var = (double)max_luma_value * max_luma_value * (source_width / 2 * source_height / 2);

    cb_psnr = get_psnr_tl26((double)cb_sse, temp_var);

    cr_psnr = get_psnr_tl26((double)cr_sse, temp_var);

    report_frame_feedback_helper(picture_number,
                                 temporal_layer_index,
                                 picture_qp,
                                 avg_qp,
                                 luma_psnr,
                                 cb_psnr,
                                 cr_psnr,
                                 (double)luma_sse / (source_width * source_height),
                                 (double)cb_sse / (source_width / 2 * source_height / 2),
                                 (double)cr_sse / (source_width / 2 * source_height / 2),
                                 luma_ssim,
                                 cb_ssim,
                                 cr_ssim,
                                 picture_stream_size);
}
