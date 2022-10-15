import ffmpeg
import numpy as np
from beartype import beartype
from misc_utils.beartypes import NumpyFloat1D
from webvtt import WebVTT, Caption

from whisper.audio import SAMPLE_RATE
from whisper.utils import format_timestamp


@beartype
def load_audio_from_bytes(audio_bytes: bytes, sr: int = SAMPLE_RATE) -> NumpyFloat1D:
    """
    based on: https://github.com/openai/whisper/blob/d18e9ea5dd2ca57c697e8e55f9e654f06ede25d0/whisper/audio.py#L22
    """
    try:
        out, _ = (
            ffmpeg.input("pipe:", threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(capture_stdout=True, capture_stderr=True, input=audio_bytes)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

@beartype
def write_webvtt_file(
    start_end_text: list[tuple[float, float, str]],
    file,
):
    # TODO: is there really no lib doing this? including this timestamp formatting
    vtt = WebVTT()
    for s,e,t in start_end_text:
        caption = Caption(
            format_timestamp(round(s * 1000)),
            format_timestamp(round(e * 1000)),
            t,
        )
        vtt.captions.append(caption)
    vtt.save(file)
