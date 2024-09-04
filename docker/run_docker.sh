#!/bin/bash

# Name of the Docker image
IMAGE_NAME="penn-figueroa-lab/lpvds"
IMAGE_TAG=latest

# Container name (optional)
#CONTAINER_NAME="my_ubuntu_container"

# Display settings
RUN_FLAGS+=(-e DISPLAY="${DISPLAY}")
RUN_FLAGS+=(-e XAUTHORITY="${XAUTHORITY}")
RUN_FLAGS+=(-v /tmp/.X11-unix:/tmp/.X11-unix:rw)

# Additional options (e.g., to run in interactive mode with a TTY)
RUN_FLAGS+=(-it)

# Run the Docker container with the specified flags
docker run --name "${CONTAINER_NAME}" "${RUN_FLAGS[@]}" "${IMAGE_NAME}:${IMAGE_TAG}"
