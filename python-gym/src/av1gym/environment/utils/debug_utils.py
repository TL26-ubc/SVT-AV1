import cv2
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

    # Re-assemble the planar buffer exactly as it came off disk/encoder
    yuv420 = np.concatenate(
        [y.flatten(), u.flatten(), v.flatten()]
    ).astype(np.uint8)

    # OpenCV expects shape (h * 3/2, w) for I420
    yuv420 = yuv420.reshape((int(h * 1.5), w))

    # Convert to BGR (or RGB if you prefer)
    bgr = cv2.cvtColor(yuv420, cv2.COLOR_YUV2BGR_I420)

    # Save
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), bgr)
