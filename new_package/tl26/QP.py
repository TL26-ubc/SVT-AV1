def calculate_adjustment(available_bit_ratio, 
                         available_frames_ratio, 
                         buff_lvl_step, 
                         active_worst_quality, 
                         OPTIMAL_BUFFER_LEVEL, 
                         CRITICAL_BUFFER_LEVEL):
    adjustment = 0

    if available_bit_ratio <= OPTIMAL_BUFFER_LEVEL:
        if available_bit_ratio > CRITICAL_BUFFER_LEVEL:
            max_adjustment = active_worst_quality if (available_bit_ratio + 20 < available_frames_ratio) else active_worst_quality // 2
            if buff_lvl_step and available_bit_ratio < available_frames_ratio + 10:
                adjustment = int(max_adjustment * (OPTIMAL_BUFFER_LEVEL - available_bit_ratio) / buff_lvl_step)
        else:
            adjustment = active_worst_quality

    return adjustment