import pytest
from beartype.door import is_bearable
from misc_utils.beartypes import NumpyFloat1D

from app.utils import load_audio_from_bytes


def test_load_audio_from_bytes(audio_file: str):
    """
    test just for promotional/educational purpose
    to show that beartype already enforces+validates that loaded audio:
      1. float32
      2. one-dimensional
      3. non-empty
    see: NumpyFloat1D
    """
    with open(audio_file, "rb") as f:
        b = f.read()
        audio = load_audio_from_bytes(b)
    assert is_bearable(audio, NumpyFloat1D)


def test_fail_load_audio_from_bytes(audio_file: str):
    b = "foo-bar".encode("utf-8")
    with pytest.raises(RuntimeError) as exc_info:
        _ = load_audio_from_bytes(b)
