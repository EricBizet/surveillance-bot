ARG DOCKER_IMAGE_TAG=11.3.1-devel-ubuntu20.04
FROM nvidia/cuda:${DOCKER_IMAGE_TAG} AS Build

ARG SG_VERSION=3.6.1
ARG DEBIAN_FRONTEND=noninteractive
RUN mkdir /SG

RUN apt-get update && apt-get install -y python3-pip python-is-python3 pip libgl1 libglib2.0-0 git python3-distutils python3-typing-extensions \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu --no-cache-dir
RUN python -m pip install typing-extensions --upgrade
RUN python -m pip install super_gradients==${SG_VERSION} --no-cache-dir

WORKDIR /SG

COPY export_onnx.py ./
RUN python export_onnx.py

FROM python:3.12.2-slim-bookworm

COPY ./requirements_run.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY --from=build ./*.onnx ./
