import re

import Levenshtein
import pytest
from starlette import status


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


def _poormans_text_normalization(hyp):
    hyp_upper_no_punct = re.sub(r"[^A-Z|\s]", "", hyp.upper())
    hyp_upper_no_punct = re.sub(r"\s+", " ", hyp_upper_no_punct)
    return hyp_upper_no_punct


@pytest.mark.parametrize(
    "task_cer",
    [
        ("transcribe", 0.03),
        ("translate", 0.03),  # does not make that much sense cause its english->english
    ],
)
def test_tasks(test_client, audio_file, transcript_reference, task_cer):
    task, max_CER = task_cer

    f = open(audio_file, "rb")
    files = {"file": (f.name, f, "multipart/form-data")}

    resp = test_client.post(
        f"/{task}",
        files=files,
    )

    assert resp.status_code == status.HTTP_200_OK
    resp_dict = resp.json()
    hyp = resp_dict["text"]
    ref = transcript_reference.upper()

    hyp_upper_no_punct = _poormans_text_normalization(hyp)
    cer = Levenshtein.distance(hyp_upper_no_punct, ref) / len(ref)
    assert cer <= max_CER


def test_classify_language(test_client, audio_file):
    f = open(audio_file, "rb")
    files = {"file": (f.name, f, "multipart/form-data")}

    resp = test_client.post(
        f"/classify_lang",
        files=files,
    )
    print(f"{resp=}")
    print(f"{resp.json()}")
    assert resp.status_code == status.HTTP_200_OK
    assert resp.json()["lang_code"] == "en"
