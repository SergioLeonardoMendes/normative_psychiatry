#!/bin/bash
set -ex
TAG=sgospm12

docker build  --network=host --tag "10.202.67.207:5000/${USER}:${TAG}" -f ./Dockerfile.preprocessing . \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  --build-arg USER=${USER}

docker push "10.202.67.207:5000/${USER}:${TAG}"