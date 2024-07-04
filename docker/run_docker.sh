#!/bin/bash

NAME=biodiversity_docker
MAIN_DIR=$(realpath "`dirname $0`/../")

cd "`dirname $0`"
NETWORK="host"

docker run -it --rm --gpus all -e PLATFORM=$ENV_PLATFORM \
    --network $NETWORK \
    -v $MAIN_DIR:/home/work \
    --shm-size 16G \
    $NAME \
    bash
