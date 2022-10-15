import re
from pprint import pprint

import Levenshtein
import pytest
from fastapi.testclient import TestClient
from starlette import status

from app.main import app


@pytest.fixture(scope="module")
def test_client() -> TestClient:
    with TestClient(app) as tc:
        yield tc


@pytest.fixture(scope="module")
def audio_file() -> str:
    return "tests/resources/LibriSpeech_dev-other_116_288046_116-288046-0011.opus"


@pytest.fixture(scope="module")
def transcript_reference() -> str:
    file = "tests/resources/LibriSpeech_dev-other_116_288046_116-288046-0011_ref.txt"
    with open(file, "r", encoding="utf-8") as f:
        ref = f.read()

    return ref


@pytest.mark.parametrize(
    "lang_error_code",
    [
        ("en", status.HTTP_200_OK),
        ("not-valid-lang", status.HTTP_422_UNPROCESSABLE_ENTITY),
    ],
)
def test_language(test_client, audio_file, lang_error_code):
    lang, error_code = lang_error_code

    f = open(audio_file, "rb")
    files = {"file": (f.name, f, "multipart/form-data")}

    resp = test_client.post("/transcribe", files=files, params={"language": lang})

    assert resp.status_code == error_code


@pytest.mark.parametrize(
    "input_error_code",
    [
        ("base", status.HTTP_200_OK),
        ("not-valid-model_name", status.HTTP_422_UNPROCESSABLE_ENTITY),
    ],
)
def test_load_model(test_client, input_error_code):
    model_name, error_code = input_error_code
    resp = test_client.get(f"/load_model/{model_name}")
    assert resp.status_code == error_code


def test_transcripe_endpoint(test_client, audio_file, transcript_reference):
    max_CER = 0.03

    f = open(audio_file, "rb")
    files = {"file": (f.name, f, "multipart/form-data")}

    resp = test_client.post(
        "/transcribe",
        files=files,
    )

    assert resp.status_code == status.HTTP_200_OK
    resp_dict = resp.json()
    hyp = resp_dict["text"]
    ref = transcript_reference.upper()

    hyp_upper_no_punct = _poormans_text_normalization(hyp)
    cer = Levenshtein.distance(hyp_upper_no_punct, ref) / len(ref)
    assert cer <= max_CER


def _poormans_text_normalization(hyp):
    hyp_upper_no_punct = re.sub(r"[^A-Z|\s]", "", hyp.upper())
    hyp_upper_no_punct = re.sub(r"\s+", " ", hyp_upper_no_punct)
    return hyp_upper_no_punct
