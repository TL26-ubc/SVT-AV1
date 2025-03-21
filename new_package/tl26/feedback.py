from .global_variables import frame_feed_backs

print("Hello SVT-AV1 from python feedback package")

class Frame_feedback:
    def __init__(self, picture_number: int, temporal_layer_index: int, qp: int, ave_qp: int,
                 psnr_y: float, psnr_u: float, psnr_v: float,
                 mse_y: float, mse_u: float, mse_v: float,
                 ssim_y: float, ssim_u: float, ssim_v: float,
                 picture_stream_size: int
                 ):
        self.picture_number = picture_number
        self.temporal_layer_index = temporal_layer_index
        self.qp = qp
        self.ave_qp = ave_qp
        self.psnr_y = psnr_y
        self.psnr_u = psnr_u
        self.psnr_v = psnr_v
        self.mse_y = mse_y
        self.mse_u = mse_u
        self.mse_v = mse_v
        self.ssim_y = ssim_y
        self.ssim_u = ssim_u
        self.ssim_v = ssim_v
        self.picture_stream_size = picture_stream_size
        
    def to_float_list(self):
        return [float(self.picture_number), float(self.temporal_layer_index), float(self.qp), float(self.ave_qp),
                float(self.psnr_y), float(self.psnr_u), float(self.psnr_v),
                float(self.mse_y), float(self.mse_u), float(self.mse_v),
                float(self.ssim_y), float(self.ssim_u), float(self.ssim_v),
                float(self.picture_stream_size)]
        
    def report(self):
        print (f"Frame_feedback(picture_number={self.picture_number}, temporal_layer_index={self.temporal_layer_index}, qp={self.qp}, ave_qp={self.ave_qp}, " + \
                f"psnr_y={self.psnr_y}, psnr_u={self.psnr_u}, psnr_v={self.psnr_v}, " + \
                f"mse_y={self.mse_y}, mse_u={self.mse_u}, mse_v={self.mse_v}, " + \
                f"ssim_y={self.ssim_y}, ssim_u={self.ssim_u}, ssim_v={self.ssim_v}, " + \
                f"picture_stream_size={self.picture_stream_size})")
        
def frame_report_feedback(picture_number: int, temporal_layer_index: int, qp: int, ave_qp: int,
                 psnr_y: float, psnr_u: float, psnr_v: float,
                 mse_y: float, mse_u: float, mse_v: float,
                 ssim_y: float, ssim_u: float, ssim_v: float,
                 picture_stream_size: int):
    frame_feedback = Frame_feedback(picture_number, temporal_layer_index, qp, ave_qp,
                    psnr_y, psnr_u, psnr_v,
                    mse_y, mse_u, mse_v,
                    ssim_y, ssim_u, ssim_v,
                    picture_stream_size)
    
    # check no previous feedback
    if frame_feedback.picture_number in frame_feed_backs:
        raise ValueError(f"Frame_feedback already exists for picture_number {frame_feedback.picture_number}")

    frame_feed_backs[frame_feedback.picture_number] = frame_feedback
    
    frame_feedback.report()
    
    return