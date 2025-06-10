#ifndef PYBRIDGE_H_
#define PYBRIDGE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "../../../Source/API/EbSvtAv1Enc.h"

typedef void (*get_deltaq_offset_cb_t)(int *, uint32_t, int32_t);

typedef void (*recv_picture_feedback_cb_t)(uint8_t *, uint32_t, uint32_t);

extern get_deltaq_offset_cb_t     get_deltaq_offset_cb;
extern recv_picture_feedback_cb_t recv_picture_feedback_cb;

void get_deltaq_offset_trampoline(int *, uint32_t, int32_t);

void recv_picture_feedback_trampoline(uint8_t *, uint32_t, uint32_t);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* PYBRIDGE_H_ */
