#!/usr/bin/env bash
#
# This script builds the mardi-ai-engine docker container from source.
#

CONTAINER_LOCAL_IMAGE="registry.pintarai.com/pintarai/mardi:latest"
echo "CONTAINER_LOCAL_IMAGE=$CONTAINER_LOCAL_IMAGE"
		
# build the container
sudo docker build -t $CONTAINER_LOCAL_IMAGE -f Dockerfile .
sudo docker push $CONTAINER_LOCAL_IMAGE