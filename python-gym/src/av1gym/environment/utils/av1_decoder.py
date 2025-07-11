from pathlib import Path
import av

class Av1OnTheFlyDecoder:
    def __init__(self):
        # libaom-av1 = av1 impl. with supported av bindings, r = decode direction
        self._decoder: av.VideoCodecContext = av.CodecContext.create('libaom-av1', 'r') # type: ignore
        self._frames: list[av.VideoFrame] = []
        self._packets: list[av.Packet] = []

    def reset(self):
        """Start a new GOP / drop reference pictures"""
        self._decoder.flush_buffers()

    def append(self, packet: av.Packet | bytes | int) -> av.VideoFrame:
        """
        Feed one encoded AV1 packet, returns the last decoded frame
        """
        if not isinstance(packet, av.Packet):
            packet = av.Packet(packet)
        self._frames = self._decoder.decode(packet)
        self._packets.append(packet)
        return self._frames[-1]

    def get_frames(self) -> list[av.VideoFrame] | None:
        """
        Get all frames currently in the decoder
        """
        return self._frames
    
    def save_video(
        self,
        output_path: str | Path,
    ) -> None:
        """
        Output the current video to a file
        """
        if not self._packets:
            raise RuntimeError("No frames have been decoded yet.")

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with av.open(out_path, "w") as container:
            container.mux(self._packets)
