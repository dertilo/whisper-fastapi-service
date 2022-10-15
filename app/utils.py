from typing import BinaryIO

import ffmpeg
import numpy as np
from beartype import beartype
from whisper.audio import SAMPLE_RATE


@beartype
def load_audio_from_bytes(audio_bytes: bytes, sr: int = SAMPLE_RATE):
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
