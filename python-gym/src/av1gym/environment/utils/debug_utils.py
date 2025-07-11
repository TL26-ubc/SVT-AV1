import av
import numpy as np
from pathlib import Path

def save_debug_frame(y: np.ndarray,
                     u: np.ndarray,
                     v: np.ndarray,
                     out_path: str | Path = "debug_frame.png") -> None:
    """
    Quickly sanity-check YUV420p planes by round-tripping them to an RGB image.

    Parameters
    ----------
    y, u, v : np.ndarray
        • y.shape == (h, w)
        • u,v.shape == (h/2, w/2)
        Must be dtype=uint8 and full-range (0-255) for 8-bit video.
    out_path : str | pathlib.Path
        Where to write the PNG (format auto-detected by OpenCV).
    """
    h, w = y.shape
    assert u.shape == v.shape == (h // 2, w // 2), "U/V plane size mismatch"

    # Allocate an empty YUV420p frame and copy each plane
    yuv_frame = av.VideoFrame(width=w, height=h, format="yuv420p")
    yuv_frame.planes[0].update(y.tobytes()) # Y
    yuv_frame.planes[1].update(u.tobytes()) # U
    yuv_frame.planes[2].update(v.tobytes()) # V

    # Convert to rgb
    rgb_frame = yuv_frame.reformat(format="rgb24")

    # Save
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    rgb_frame.to_image().save(Path(out_path))
