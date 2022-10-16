# Whisper FastAPI Service [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![bear-ified](https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg)](https://beartype.readthedocs.io)
OpenAI's [Whisper](https://github.com/openai/whisper/) dockerized and put behind [FastAPI](https://fastapi.tiangolo.com/)

## features
* transcribe/translate via fastapi's [UploadFile](https://fastapi.tiangolo.com/tutorial/request-files/#uploadfile) [form-data](https://fastapi.tiangolo.com/tutorial/request-files/#what-is-form-data)
  * as response get a json 
  * or get a vtt-file
* load whisper model
  1. via environment variable to docker-container: `docker run -e MODEL_NAME=base ...`
  2. get-request: `curl http://localhost:2700/load_model/large`
     * gives you response like this: `{"loaded_model":"https://openaipublic.azureedge.net/main/whisper/models/<some-hash>/large.pt"}`
* [docker-image](https://hub.docker.com/repository/docker/dertilo/whisper-fastapi-service/general)

## TL;DR
1. run docker-container
```commandline
docker run --rm -p 2700:2700 dertilo/whisper-fastapi-service:latest
```
2. transcribe files of "almost any format": `wav,flac,mp3,opus,mp4,...`
   * either goto: `localhost:2700/docs`
   * OR curl it: `curl -F 'file=@<some-where>/<your_file>' http://localhost:2700/transcribe`


## run service locally
```commandline
# run service
pip install -r requirements.txt
python app/main.py

# in another terminal make request
curl -F 'file=@tests/resources/LibriSpeech_dev-other_116_288046_116-288046-0011.opus' http://localhost:2700/transcribe
```
* response json looks like this: 
```json
{
    "text": " Not having the courage or the industry of our neighbor, who works like a busy bee in the world of men and books, searching with the sweat of his brow for the real bread of life, waiting the open page of for him with his tears, pushing into the wee hours of the night his quest, animated by the fairest of all loves, the love of truth. We ease our own indolent conscience by calling him names.",
    "segments": [
        {
            "id": "0",
            "seek": 0,
            "start": 0,
            "end": 6,
            "text": " Not having the courage or the industry of our neighbor, who works like a busy bee in the world of men and books,",
            "tokens": [
                50364,
                1726,
                1419,
      ...

              }
}
```

## build & run docker-image/container
```commandline
docker build -t whisper-fastapi-service .

docker run --rm -p 2700:2700 whisper-fastapi-service
```

# TODO
* use ONNX-models via OpenVino, [see](https://github.com/openai/whisper/discussions/208)
* GPU-docker-image