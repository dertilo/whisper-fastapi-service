FROM python:3.9-bullseye

WORKDIR /code

RUN export DEBIAN_FRONTEND=noninteractive \
  && apt-get -qq update \
  && apt-get -qq upgrade \
  && apt-get -qq install --no-install-recommends \
  sox \
  ffmpeg \
  libsndfile1 \
  && rm -rf /var/lib/apt/lists/*

ENV PIP_INSTALL="pip install --no-cache-dir --upgrade"

COPY requirements.txt ./
RUN $PIP_INSTALL -r requirements.txt

COPY app ./app

EXPOSE 2700/tcp

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "2700"]
