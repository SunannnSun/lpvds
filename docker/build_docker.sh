#!/bin/bash
LOCAL_BASE_IMAGE=false
BASE_TAG=humble

IMAGE_NAME=penn-figueroa-lab/lpvds
IMAGE_TAG=latest

REMOTE_SSH_PORT=1003
SERVE_REMOTE=false

HELP_MESSAGE="Usage: build.sh [-p] [-r]
Options:
  -d, --development      Only target the dependencies layer to prevent
                         sources from being built or tested

  -r, --rebuild          Rebuild the image(s) using the docker
                         --no-cache option

  -v, --verbose          Use the verbose option during the building
                         process
"

BUILD_FLAGS=(--build-arg BASE_TAG="${BASE_TAG}")
while [[ $# -gt 0 ]]; do
  opt="$1"
  case $opt in
    -d|--development) BUILD_FLAGS+=(--target dependencies) ; IMAGE_TAG=development ; shift ;;
    -r|--rebuild) BUILD_FLAGS+=(--no-cache) ; shift ;;
    -v|--verbose) BUILD_FLAGS+=(--progress=plain) ; shift ;;
    -h|--help) echo "${HELP_MESSAGE}" ; exit 0 ;;
    *) echo 'Error in command line parsing' >&2
       echo -e "\n${HELP_MESSAGE}"
       exit 1
  esac
done

# if [ "${LOCAL_BASE_IMAGE}" == true ]; then
#   BUILD_FLAGS+=(--build-arg BASE_IMAGE=aica-technology/ros2-modulo)
# else
#   docker pull ghcr.io/aica-technology/ros2-modulo:"${BASE_TAG}"
# fi

DOCKER_BUILDKIT=1 docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" "${BUILD_FLAGS[@]}" -f ./docker/Dockerfile . || exit 1
