#include "enc_callbacks.h"
#include "../Lib/Codec/coding_unit.h"

#ifdef SVT_ENABLE_USER_CALLBACKS
PluginCallbacks plugin_cbs;

/**********************************
* Plugin callback registration
**********************************/
EB_API EbErrorType svt_av1_enc_set_callbacks(const PluginCallbacks *cbs)
{
    if (!cbs)
        return EB_ErrorBadParameter;

    plugin_cbs = *cbs;
    return EB_ErrorNone;
}
#endif