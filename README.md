# whisper-fastapi-service [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
OpenAI's Whisper dockerized and put behind FastAPI

### run locally
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

### build & run docker-image/container
```commandline
docker build -t whisper-fastapi-service .

docker run --rm -p 2700:2700 whisper-fastapi-service
```