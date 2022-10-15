from dataclasses import dataclass


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
