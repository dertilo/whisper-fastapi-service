import os
from dataclasses import dataclass
from enum import Enum
from io import StringIO, BytesIO
from typing import Optional, Union

import starlette
import uvicorn
from beartype import beartype
from starlette.responses import FileResponse, StreamingResponse

import whisper
from beartype.door import is_bearable
from fastapi import FastAPI, UploadFile, Query

from app.models import WhisperResponse
from whisper import Whisper
from whisper.tokenizer import LANGUAGES

from app.utils import load_audio_from_bytes, write_webvtt_file

app = FastAPI()
inferencer: Optional[Whisper] = None

MODEL_NAME = os.environ.get("MODEL_NAME", "base")
assert MODEL_NAME in whisper._MODELS.keys()
LANGUAGE_CODES = sorted(list(LANGUAGES.keys()))


@app.on_event("startup")
def startup_event():
    global inferencer
    inferencer = whisper.load_model(MODEL_NAME)


class TaskName(str, Enum):
    # see: https://github.com/openai/whisper/blob/d18e9ea5dd2ca57c697e8e55f9e654f06ede25d0/whisper/transcribe.py#L260
    transcribe = "transcribe"
    translate = "translate"


@app.post("/vtt-file/{task_name}")
async def vtt_file(
    task_name: TaskName,
    file: UploadFile,
    language: Union[str, None] = Query(
        default=None, regex=f"^(?:{'|'.join(LANGUAGE_CODES)})$"
    ),
):
    result = await _load_and_predict(task_name, inferencer, file, language)

    # inspired by: https://cloudbytes.dev/snippets/received-return-a-file-from-in-memory-buffer-using-fastapi

    vtt_file = StringIO()
    write_webvtt_file(
        start_end_text=[
            (seg["start"], seg["end"], seg["text"]) for seg in result["segments"]
        ],
        file=vtt_file,
    )
    vtt_file.seek(0)
    return StreamingResponse(
        vtt_file,
        media_type="text/plain",
    )


@dataclass
class ClassifyLangResponse:
    lang_name: str  # lang instead of language cause this word is "too difficult"! -> langauge
    lang_code: str


# this must come before the "task" endpoint, cause they are in same "root"-path -> "/"
@app.post("/classify_lang", response_model=ClassifyLangResponse)
async def classify_lang(
    file: UploadFile,
):
    global inferencer
    data_bytes = await file.read()  # in synchronous context do:  file.file.read()
    audio = load_audio_from_bytes(data_bytes)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(inferencer.device)

    _, probs = inferencer.detect_language(mel)  # yes its blocking! who cares?
    lang_code = max(probs, key=probs.get)
    # TODO: maybe return dict lang_code : prob?
    return {"lang_name": LANGUAGES[lang_code], "lang_code": lang_code}


@app.post("/{task_name}", response_model=WhisperResponse)
async def predict(
    task_name: TaskName,
    file: UploadFile,
    language: Union[str, None] = Query(
        default=None, regex=f"^(?:{'|'.join(LANGUAGE_CODES)})$"
    ),
):
    global inferencer

    # Type narrowing via assert to keep mypy happy and just for fun using beartype here, cause I like the bear!
    assert is_bearable(inferencer, Whisper)

    result = await _load_and_predict(task_name, inferencer, file, language)

    return result


_UploadFile = Union[UploadFile, starlette.datastructures.UploadFile]


@beartype
async def _load_and_predict(
    task_name: str, inferencer: Whisper, file: _UploadFile, language: Optional[str]
):
    data_bytes = await file.read()  # in synchronous context do:  file.file.read()
    audio = load_audio_from_bytes(data_bytes)
    # voluntarily blocking the async event loop cause heavy-lifting (cpu-bound) stuff in ThreadPool makes no sense
    result = inferencer.transcribe(audio, task=task_name, language=language)
    return result


class TempEnum(str, Enum):
    pass


# dynamic enum creation see: https://github.com/tiangolo/fastapi/issues/13
WhisperModelName = TempEnum("TypeEnum", {k: k for k in whisper._MODELS.keys()})


@app.get("/load_model/{model_name}")
async def predict(
    model_name: WhisperModelName,
):
    global inferencer
    inferencer = whisper.load_model(MODEL_NAME)
    return {"loaded_model": whisper._MODELS[model_name]}


if __name__ == "__main__":  # pragma: no cover
    """
    just for debugging
    """
    import sys

    sys.path.append(".")
    uvicorn.run(
        "app.main:app", host="0.0.0.0", port=2700, log_level="debug", reload=True
    )
