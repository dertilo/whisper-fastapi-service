FROM python:3.9-bullseye as build

WORKDIR /code

RUN export DEBIAN_FRONTEND=noninteractive \
  && apt-get -qq update \
  && apt-get -qq upgrade \
  && apt-get -qq install --no-install-recommends \
  python3-venv \
  sox \
  ffmpeg \
  libsndfile1 \
  && rm -rf /var/lib/apt/lists/*

ENV PATH="/venv/bin:$PATH"
ENV PIP_INSTALL="/venv/bin/pip install --no-cache-dir --upgrade"
RUN python3 -m venv /venv && $PIP_INSTALL pip packaging setuptools

COPY requirements.txt ./
COPY requirements_test.txt ./
RUN $PIP_INSTALL -r requirements.txt

COPY app ./app
COPY tests ./tests

RUN $PIP_INSTALL -r requirements_test.txt
RUN python -m pytest .

# ----------- PRODUCTION STAGE ----------------------------------
FROM python:3.9-slim-bullseye AS prod
ENV PATH="/venv/bin:$PATH"

WORKDIR /code

EXPOSE 2700/tcp

LABEL maintainer="Tilo Himmelsbach"
COPY --from=build /venv /venv
COPY --from=build /code/app /code/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "2700"]
