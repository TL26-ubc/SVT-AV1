#include "tl26_flags.h"
#include "tl26_feedback.h"
#include "tl26_python_thread.h"

#ifdef TL26_RL
static inline double get_psnr_tl26(double sse, double max) {
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
    submit_frame_feedback_request(
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
}

void report_frame_feedback(EbBufferHeaderType *header_ptr, EbConfig *app_cfg) {
    uint32_t max_luma_value = (app_cfg->config.encoder_bit_depth == 8) ? 255 : 1023;
    uint8_t temporal_layer_index;
    uint32_t avg_qp;
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
    temporal_layer_index = header_ptr->temporal_layer_index;
    picture_qp = header_ptr->qp;
    avg_qp = header_ptr->avg_qp;
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
#else
void report_frame_feedback(EbBufferHeaderType *header_ptr, EbConfig *app_cfg) {
    (void)header_ptr;
    (void)app_cfg;
}
#endif

EbErrorType svt_aom_ssim_calculations_sb(PictureControlSet *pcs, SequenceControlSet *scs, SuperBlock *sb, Bool free_memory) {
    Bool is_16bit = (scs->static_config.encoder_bit_depth > EB_EIGHT_BIT);
    SuperBlock *sb;
    uint32_t sb_width  = MIN(scs->sb_size, pcs->ppcs->aligned_width  - sb->org_x);
    uint32_t sb_height = MIN(scs->sb_size, pcs->ppcs->aligned_height - sb->org_y);

    const uint32_t ss_x = scs->subsampling_x;
    const uint32_t ss_y = scs->subsampling_y;

    EbPictureBufferDesc *recon_ptr;
    EbPictureBufferDesc *input_pic = (EbPictureBufferDesc *)pcs->ppcs->enhanced_unscaled_pic;
    svt_aom_get_recon_pic(pcs, &recon_ptr, is_16bit);
    // upscale recon if resized
    EbPictureBufferDesc *upscaled_recon = NULL;
    Bool                 is_resized = recon_ptr->width != input_pic->width || recon_ptr->height != input_pic->height;
    if (is_resized) {
        superres_params_type spr_params = {input_pic->width, input_pic->height, 0};
        svt_aom_downscaled_source_buffer_desc_ctor(&upscaled_recon, recon_ptr, spr_params);
        svt_aom_resize_frame(recon_ptr,
                             upscaled_recon,
                             scs->static_config.encoder_bit_depth,
                             av1_num_planes(&scs->seq_header.color_config),
                             ss_x,
                             ss_y,
                             recon_ptr->packed_flag,
                             PICTURE_BUFFER_DESC_FULL_MASK,
                             0); // is_2bcompress
        recon_ptr = upscaled_recon;
    }

    if (!is_16bit) {
        EbByte input_buffer;
        EbByte recon_coeff_buffer;

        EbByte buffer_y;
        EbByte buffer_cb;
        EbByte buffer_cr;

        double luma_ssim = 0.0;
        double cb_ssim   = 0.0;
        double cr_ssim   = 0.0;

        // if current source picture was temporally filtered, use an alternative buffer which stores
        // the original source picture
        if (pcs->ppcs->do_tf == TRUE) {
            assert(pcs->ppcs->save_source_picture_width == input_pic->width &&
                   pcs->ppcs->save_source_picture_height == input_pic->height);
            buffer_y  = pcs->ppcs->save_source_picture_ptr[0];
            buffer_cb = pcs->ppcs->save_source_picture_ptr[1];
            buffer_cr = pcs->ppcs->save_source_picture_ptr[2];
        } else {
            buffer_y  = input_pic->buffer_y;
            buffer_cb = input_pic->buffer_cb;
            buffer_cr = input_pic->buffer_cr;
        }

        recon_coeff_buffer = &((recon_ptr->buffer_y)[(recon_ptr->org_x + sb->org_x) + 
            (recon_ptr->org_y + sb->org_y) * recon_ptr->stride_y]);
        input_buffer       = &(buffer_y[(input_pic->org_x + sb->org_x) + 
            (input_pic->org_y + sb->org_y) * input_pic->stride_y]);
        luma_ssim          = aom_ssim2(input_buffer,
                              input_pic->stride_y,
                              recon_coeff_buffer,
                              recon_ptr->stride_y,
                              sb_width,
                              sb_height);

        recon_coeff_buffer = &((recon_ptr->buffer_cb)[(recon_ptr->org_x + sb->org_x) / 2 + 
            (recon_ptr->org_y + sb->org_y) / 2 * recon_ptr->stride_cb]);
        input_buffer = &(buffer_cb[(input_pic->org_x + sb->org_x) / 2 + 
            (input_pic->org_y + sb->org_y) / 2 * input_pic->stride_cb]);
        cb_ssim      = aom_ssim2(input_buffer,
                            input_pic->stride_cb,
                            recon_coeff_buffer,
                            recon_ptr->stride_cb,
                            sb_width,
                            sb_height);

        recon_coeff_buffer = &((recon_ptr->buffer_cr)[(recon_ptr->org_x + sb->org_x) / 2 + 
                (recon_ptr->org_y + sb->org_y) / 2 * recon_ptr->stride_cr]);
        input_buffer = &(buffer_cr[(input_pic->org_x + sb->org_x) / 2 + 
            (input_pic->org_y + sb->org_y) / 2 * input_pic->stride_cr]);
        cr_ssim      = aom_ssim2(input_buffer,
                            input_pic->stride_cr,
                            recon_coeff_buffer,
                            recon_ptr->stride_cr,
                            sb_width,
                            sb_height);

        // TODO save calculation somewhere
        // pcs->ppcs->luma_ssim = luma_ssim;
        // pcs->ppcs->cb_ssim   = cb_ssim;
        // pcs->ppcs->cr_ssim   = cr_ssim;

        if (free_memory && pcs->ppcs->do_tf == TRUE) {
            EB_FREE_ARRAY(buffer_y);
            EB_FREE_ARRAY(buffer_cb);
            EB_FREE_ARRAY(buffer_cr);
        }
    } else {
        EbByte    input_buffer;
        uint16_t *recon_coeff_buffer;

        double luma_ssim = 0.0;
        double cb_ssim   = 0.0;
        double cr_ssim   = 0.0;

        if (scs->ten_bit_format == 1) {
            /* This format is not supported on this version, we should never get here */
            fprintf(stderr, "We should not be encoding with ten bit format\n");
            exit(1);
        } else {
            recon_coeff_buffer = (uint16_t *)(&(
                (recon_ptr->buffer_y)[((recon_ptr->org_x + sb->org_x) << is_16bit) +
                                      ((recon_ptr->org_y + sb->org_y) << is_16bit) * recon_ptr->stride_y]));

            // if current source picture was temporally filtered, use an alternative buffer which stores
            // the original source picture
            EbByte buffer_y, buffer_bit_inc_y;
            EbByte buffer_cb, buffer_bit_inc_cb;
            EbByte buffer_cr, buffer_bit_inc_cr;
            int    bd, shift;

            if (pcs->ppcs->do_tf == TRUE) {
                assert(pcs->ppcs->save_source_picture_width == input_pic->width &&
                       pcs->ppcs->save_source_picture_height == input_pic->height);
                buffer_y          = pcs->ppcs->save_source_picture_ptr[0];
                buffer_bit_inc_y  = pcs->ppcs->save_source_picture_bit_inc_ptr[0];
                buffer_cb         = pcs->ppcs->save_source_picture_ptr[1];
                buffer_bit_inc_cb = pcs->ppcs->save_source_picture_bit_inc_ptr[1];
                buffer_cr         = pcs->ppcs->save_source_picture_ptr[2];
                buffer_bit_inc_cr = pcs->ppcs->save_source_picture_bit_inc_ptr[2];
            } else {
                uint32_t height_y  = (uint32_t)(input_pic->height + input_pic->org_y + input_pic->origin_bot_y);
                uint32_t height_uv = (uint32_t)((input_pic->height + input_pic->org_y + input_pic->origin_bot_y) >>
                                                ss_y);

                uint8_t *uncompressed_pics[3];
                EB_MALLOC_ARRAY(uncompressed_pics[0], pcs->ppcs->enhanced_unscaled_pic->luma_size);
                EB_MALLOC_ARRAY(uncompressed_pics[1], pcs->ppcs->enhanced_unscaled_pic->chroma_size);
                EB_MALLOC_ARRAY(uncompressed_pics[2], pcs->ppcs->enhanced_unscaled_pic->chroma_size);

                svt_c_unpack_compressed_10bit(input_pic->buffer_bit_inc_y,
                                              input_pic->stride_bit_inc_y / 4,
                                              uncompressed_pics[0],
                                              input_pic->stride_bit_inc_y,
                                              height_y);
                // U
                svt_c_unpack_compressed_10bit(input_pic->buffer_bit_inc_cb,
                                              input_pic->stride_bit_inc_cb / 4,
                                              uncompressed_pics[1],
                                              input_pic->stride_bit_inc_cb,
                                              height_uv);
                // V
                svt_c_unpack_compressed_10bit(input_pic->buffer_bit_inc_cr,
                                              input_pic->stride_bit_inc_cr / 4,
                                              uncompressed_pics[2],
                                              input_pic->stride_bit_inc_cr,
                                              height_uv);

                buffer_y          = input_pic->buffer_y;
                buffer_bit_inc_y  = uncompressed_pics[0];
                buffer_cb         = input_pic->buffer_cb;
                buffer_bit_inc_cb = uncompressed_pics[1];
                buffer_cr         = input_pic->buffer_cr;
                buffer_bit_inc_cr = uncompressed_pics[2];
            }

            bd    = 10;
            shift = 0; // both input and output are 10 bit (bitdepth - input_bd)

            input_buffer                = &((buffer_y)[(input_pic->org_x + sb->org_x) + (input_pic->org_y + sb->org_y) * input_pic->stride_y]);
            EbByte input_buffer_bit_inc = &(
                (buffer_bit_inc_y)[(input_pic->org_x + sb->org_x) + (input_pic->org_y + sb->org_y) * input_pic->stride_bit_inc_y]);
            luma_ssim = aom_highbd_ssim2(input_buffer,
                                         input_pic->stride_y,
                                         input_buffer_bit_inc,
                                         input_pic->stride_bit_inc_y,
                                         recon_coeff_buffer,
                                         recon_ptr->stride_y,
                                         scs->max_input_luma_width,
                                         scs->max_input_luma_height,
                                         bd,
                                         shift);

            recon_coeff_buffer   = (uint16_t *)(&(
                (recon_ptr->buffer_cb)[((recon_ptr->org_x + sb->org_x) << is_16bit) / 2 +
                                       ((recon_ptr->org_y + sb->org_y) << is_16bit) / 2 * recon_ptr->stride_cb]));
            input_buffer         = &((buffer_cb)[(input_pic->org_x + sb->org_x) / 2 + (input_pic->org_y + sb->org_y) / 2 * input_pic->stride_cb]);
            input_buffer_bit_inc = &(
                (buffer_bit_inc_cb)[(input_pic->org_x + sb->org_x) / 2 + (input_pic->org_y + sb->org_y) / 2 * input_pic->stride_bit_inc_cb]);
            cb_ssim = aom_highbd_ssim2(input_buffer,
                                       input_pic->stride_cb,
                                       input_buffer_bit_inc,
                                       input_pic->stride_bit_inc_cb,
                                       recon_coeff_buffer,
                                       recon_ptr->stride_cb,
                                       scs->chroma_width,
                                       scs->chroma_height,
                                       bd,
                                       shift);

            recon_coeff_buffer   = (uint16_t *)(&(
                (recon_ptr->buffer_cr)[((recon_ptr->org_x + sb->org_x) << is_16bit) / 2 +
                                       ((recon_ptr->org_y + sb->org_y) << is_16bit) / 2 * recon_ptr->stride_cr]));
            input_buffer         = &((buffer_cr)[(input_pic->org_x + sb->org_x) / 2 + 
                (input_pic->org_y + sb->org_y) / 2 * input_pic->stride_cr]);
            input_buffer_bit_inc = &(
                (buffer_bit_inc_cr)[(input_pic->org_x + sb->org_x) / 2 + 
                    (input_pic->org_y + sb->org_y) / 2 * input_pic->stride_bit_inc_cr]);
            cr_ssim = aom_highbd_ssim2(input_buffer,
                                       input_pic->stride_cr,
                                       input_buffer_bit_inc,
                                       input_pic->stride_bit_inc_cr,
                                       recon_coeff_buffer,
                                       recon_ptr->stride_cr,
                                       scs->chroma_width,
                                       scs->chroma_height,
                                       bd,
                                       shift);

            // TODO save calculation somewhere
            // pcs->ppcs->luma_ssim = luma_ssim;
            // pcs->ppcs->cb_ssim   = cb_ssim;
            // pcs->ppcs->cr_ssim   = cr_ssim;

            if (free_memory && pcs->ppcs->do_tf == TRUE) {
                EB_FREE_ARRAY(buffer_y);
                EB_FREE_ARRAY(buffer_cb);
                EB_FREE_ARRAY(buffer_cr);
                EB_FREE_ARRAY(buffer_bit_inc_y);
                EB_FREE_ARRAY(buffer_bit_inc_cb);
                EB_FREE_ARRAY(buffer_bit_inc_cr);
            }
            if (pcs->ppcs->do_tf == FALSE) {
                EB_FREE_ARRAY(buffer_bit_inc_y);
                EB_FREE_ARRAY(buffer_bit_inc_cb);
                EB_FREE_ARRAY(buffer_bit_inc_cr);
            }
        }
    }
    EB_DELETE(upscaled_recon);
    return EB_ErrorNone;
}