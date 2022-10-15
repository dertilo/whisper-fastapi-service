import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import uvicorn
import whisper
from fastapi import FastAPI, UploadFile
from whisper import Whisper
from whisper.tokenizer import LANGUAGES

from app.utils import load_audio_from_bytes

app = FastAPI()
inferencer: Optional[Whisper] = None

MODEL_NAME = os.environ.get("MODEL_NAME", "base")
LANGUAGE_CODES = sorted(list(LANGUAGES.keys()))


@app.on_event("startup")
def startup_event():
    global inferencer
    inferencer = whisper.load_model(MODEL_NAME)


class TaskName(str, Enum):
    # see: https://github.com/openai/whisper/blob/d18e9ea5dd2ca57c697e8e55f9e654f06ede25d0/whisper/transcribe.py#L260
    transcribe = "transcribe"
    translate = "translate"


@dataclass
class WhisperSegment:
    id: str
    seek: int  # whats that?
    start: int
    end: int
    text: str
    tokens: list[int]  # see: whisper's DecodingResult
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


@dataclass
class WhisperResponse:
    text: str
    segments: list[WhisperSegment]
    language: str


@app.post("/{task_name}", response_model=WhisperResponse)
async def predict(
    task_name: TaskName,
    file: UploadFile,
):
    global inferencer
    assert isinstance(inferencer, Whisper)  # Type narrowing to keep mypy happy

    # data_bytes = file.file.read() # if in synchronous context otherwise just file
    data_bytes = await file.read()  # if in Asynchronous context
    audio = load_audio_from_bytes(data_bytes)
    # voluntarily blocking the async event loop cause heavy-lifting (cpu-bound) stuff in ThreadPool makes no sense
    result = inferencer.transcribe(audio, task=task_name)

    return result


if __name__ == "__main__":
    """
    just for debugging
    """
    import sys

    sys.path.append(".")
    uvicorn.run(
        "app.main:app", host="0.0.0.0", port=2700, log_level="debug", reload=True
    )