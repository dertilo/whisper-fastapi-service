import pytest
from fastapi.testclient import TestClient
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
