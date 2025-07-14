from pathlib import Path
import av

class Av1OnTheFlyDecoder:
    def __init__(self):
        # libaom-av1 = av1 impl. with supported av bindings, r = decode direction
        self._decoder: av.VideoCodecContext = av.CodecContext.create('libaom-av1', 'r') # type: ignore
        self._frames: list[av.VideoFrame] = []

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
        if not self._frames:
            raise RuntimeError("No frames have been decoded yet.")

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with av.open(str(out_path), "w") as container:
            # Create output stream with same properties as decoder
            stream: av.VideoStream = container.add_stream('av1') # type: ignore
            
            # Set stream parameters from the first frame
            if self._frames:
                first_frame = self._frames[0]
                stream.width = first_frame.width
                stream.height = first_frame.height
                stream.pix_fmt = first_frame.format.name
                
            # Encode and mux all frames
            for frame in self._frames:
                for packet in stream.encode(frame):
                    container.mux(packet)
                    
            # Flush any remaining packets
            for packet in stream.encode():
                container.mux(packet)