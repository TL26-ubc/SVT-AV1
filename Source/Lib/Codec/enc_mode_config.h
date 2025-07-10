#include <stdio.h>
#include <stdlib.h>

#include "pcs.h"
#include "resize.h"
#include "enc_dec_process.h"
#include "pd_process.h"
#include "pic_buffer_desc.h"

uint16_t svt_aom_get_max_can_count(EncMode enc_mode);
void     svt_aom_md_pme_search_controls(ModeDecisionContext *ctx, uint8_t md_pme_level);
void     svt_aom_set_inter_intra_ctrls(ModeDecisionContext *ctx, uint8_t inter_intra_level);

void    svt_aom_set_txt_controls(ModeDecisionContext *ctx, uint8_t txt_level);
void    svt_aom_set_obmc_controls(ModeDecisionContext *ctx, uint8_t obmc_mode);
void    svt_aom_set_wm_controls(ModeDecisionContext *ctx, uint8_t wm_level);
uint8_t svt_aom_set_nic_controls(ModeDecisionContext *ctx, uint8_t nic_level);
uint8_t svt_aom_set_chroma_controls(ModeDecisionContext *ctx, uint8_t uv_level);
uint8_t svt_aom_get_update_cdf_level(EncMode enc_mode, SliceType is_islice, uint8_t is_base);
#if TUNE_M0_3
uint8_t svt_aom_get_chroma_level(EncMode enc_mode, const uint8_t is_islice);
#else
uint8_t svt_aom_get_chroma_level(EncMode enc_mode);
#endif
uint8_t svt_aom_get_bypass_encdec(EncMode enc_mode, uint8_t encoder_bit_depth);
#if OPT_REMOVE_NIC_QP_BANDS
#if OPT_ALLINTRA_STILLIMAGE
uint8_t svt_aom_get_nic_level(SequenceControlSet *scs, EncMode enc_mode, uint8_t is_base, bool rtc_tune);
#else
uint8_t svt_aom_get_nic_level(EncMode enc_mode, uint8_t is_base, bool rtc_tune);
#endif
#else
uint8_t svt_aom_get_nic_level(EncMode enc_mode, uint8_t is_base, uint32_t qp, uint8_t seq_qp_mod, bool rtc_tune);
#endif
#if !OPT_DEPTHS_CTRL
void svt_aom_set_depth_ctrls(PictureControlSet *pcs, ModeDecisionContext *ctx, uint8_t depth_level);
#endif
uint8_t svt_aom_get_enable_me_16x16(EncMode enc_mode);
bool    svt_aom_is_ref_same_size(PictureControlSet *pcs, uint8_t list_idx, uint8_t ref_idx);
uint8_t svt_aom_get_enable_me_8x8(EncMode enc_mode, bool rtc_tune, EbInputResolution input_resolution);
void    svt_aom_sig_deriv_mode_decision_config(SequenceControlSet *scs, PictureControlSet *pcs);
void    svt_aom_sig_deriv_block(PictureControlSet *pcs, ModeDecisionContext *ctx);
void    svt_aom_sig_deriv_pre_analysis_pcs(PictureParentControlSet *pcs);
void    svt_aom_sig_deriv_pre_analysis_scs(SequenceControlSet *scs);
#if CLN_MISC
void svt_aom_sig_deriv_multi_processes(SequenceControlSet *scs, PictureParentControlSet *pcs);
#else
void svt_aom_sig_deriv_multi_processes(SequenceControlSet *scs, PictureParentControlSet *pcs,
                                       PictureDecisionContext *context_ptr);
#endif
void svt_aom_sig_deriv_me_tf(PictureParentControlSet *pcs, MeContext *me_ctx);

void svt_aom_sig_deriv_enc_dec_light_pd1(PictureControlSet *pcs, ModeDecisionContext *ctx);
void svt_aom_sig_deriv_enc_dec_light_pd0(SequenceControlSet *scs, PictureControlSet *pcs, ModeDecisionContext *ctx);
void svt_aom_sig_deriv_enc_dec_common(SequenceControlSet *scs, PictureControlSet *pcs, ModeDecisionContext *ctx);

void svt_aom_sig_deriv_me(SequenceControlSet *scs, PictureParentControlSet *pcs, MeContext *me_ctx);

void    svt_aom_sig_deriv_enc_dec(SequenceControlSet *scs, PictureControlSet *pcs, ModeDecisionContext *ctx);
uint8_t svt_aom_derive_gm_level(PictureParentControlSet *pcs, bool super_res_off);

void svt_aom_set_gm_controls(PictureParentControlSet *pcs, uint8_t gm_level);
#if OPT_ALLINTRA_STILLIMAGE_2
uint8_t svt_aom_get_enable_sg(EncMode enc_mode, uint8_t input_resolution, uint8_t fast_decode, bool avif);
uint8_t svt_aom_get_enable_restoration(EncMode enc_mode, int8_t config_enable_restoration, uint8_t input_resolution,
#if TUNE_RTC_M8
                                       uint8_t fast_decode, bool avif, bool allintra, bool rtc_tune);
#else
                                       uint8_t fast_decode, bool avif, bool allintra);
#endif
#else
uint8_t svt_aom_get_enable_sg(EncMode enc_mode, uint8_t input_resolution, uint8_t fast_decode);
uint8_t svt_aom_get_enable_restoration(EncMode enc_mode, int8_t config_enable_restoration, uint8_t input_resolution,
                                       uint8_t fast_decode);
#endif
void svt_aom_set_dist_based_ref_pruning_controls(ModeDecisionContext *ctx, uint8_t dist_based_ref_pruning_level);

bool svt_aom_get_disallow_4x4(EncMode enc_mode, uint8_t is_base);
#if OPT_RTC_B8
bool    svt_aom_get_disallow_8x8(EncMode enc_mode, bool rtc_tune, uint32_t screen_content_mode);
uint8_t svt_aom_get_nsq_geom_level(EncMode enc_mode, uint8_t is_base, InputCoeffLvl coeff_lvl, bool rtc_tune);
#else
uint8_t svt_aom_get_nsq_geom_level(EncMode enc_mode, uint8_t is_base, InputCoeffLvl coeff_lvl);
#endif
uint8_t svt_aom_get_nsq_search_level(PictureControlSet *pcs, EncMode enc_mode, InputCoeffLvl coeff_lvl, uint32_t qp);
uint8_t get_inter_compound_level(EncMode enc_mode);
uint8_t get_filter_intra_level(EncMode enc_mode);
uint8_t svt_aom_get_inter_intra_level(EncMode enc_mode, uint8_t is_base, uint8_t transition_present);
#if OPT_OBMC
uint8_t svt_aom_get_obmc_level(EncMode enc_mode, uint32_t qp, uint8_t seq_qp_mod);
#else
uint8_t svt_aom_get_obmc_level(EncMode enc_mode, uint32_t qp, uint8_t is_base, uint8_t seq_qp_mod);
#endif
void svt_aom_set_nsq_geom_ctrls(ModeDecisionContext *ctx, uint8_t nsq_geom_level, uint8_t *allow_HVA_HVB,
                                uint8_t *allow_HV4, uint8_t *min_nsq_bsize);
#if !CLN_REMOVE_PSQ_FEAT
void svt_aom_set_nsq_search_ctrls(PictureControlSet *pcs, ModeDecisionContext *ctx, uint8_t nsq_search_level,
                                  uint8_t resolution);
#endif
uint8_t svt_aom_get_tpl_synthesizer_block_size(int8_t tpl_level, uint32_t picture_width, uint32_t picture_height);

void svt_aom_set_mfmv_config(SequenceControlSet *scs);
#if OPT_USE_EXP_HME_ME
#if TUNE_MR_2
#if OPT_ALLINTRA_STILLIMAGE
void svt_aom_get_qp_based_th_scaling_factors(bool enable_qp_based_th_scaling, uint32_t *ret_q_weight,
                                             uint32_t *ret_q_weight_denom, uint32_t qp);
#else
void svt_aom_get_qp_based_th_scaling_factors(SequenceControlSet *scs, uint32_t *ret_q_weight,
                                             uint32_t *ret_q_weight_denom, uint32_t qp);
#endif
#else
void svt_aom_get_qp_based_th_scaling_factors(uint32_t qp, uint32_t *ret_q_weight, uint32_t *ret_q_weight_denom);
#endif
#endif
