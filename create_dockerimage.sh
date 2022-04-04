#!/bin/bash
#
# Script to build the Docker image to train the models.
#
# $ create_dockerimage.sh

set -ex
TAG=normativepsychiatry

docker build --network=host --tag "10.202.67.207:5000/${USER}:${TAG}" -f ./Dockerfile . \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  --build-arg USER=${USER}

#docker push "10.202.67.207:5000/${USER}:${TAG}"