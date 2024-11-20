#!/bin/bash

# Check if an image name and type were provided as arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <image_name> <type: gpu|cpu>"
    exit 1
fi

# Assign arguments to variables
IMAGE_NAME="$1-$2"
TYPE="$2"
REGISTRY="$3"

arch=$(uname -m)
if [ "$arch" = "x86_64" ]; then
    PLATFORM=""
elif [ "$arch" = "arm64" ]; then
    PLATFORM="--platform linux/amd64"
else
    PLATFORM=""
fi

latest_tag=$(docker images --format "{{.Tag}}" $IMAGE_NAME | sort -V | tail -n 1)

echo "Bumping $IMAGE_NAME"

if [ -z "$latest_tag" ]; then
    echo "No tags found for image $IMAGE_NAME. Setting initial version to 1.0.0."
    new_version="1.0.0"
else
    echo "Latest version found: $latest_tag"

    # Split the latest tag into major, minor, and patch parts
    IFS='.' read -r major minor patch <<< "$latest_tag"

    # Increment the patch version
    patch=$((patch + 1))

    # Create the new version string
    new_version="$major.$minor.$patch"
    echo "Bumped version to: $new_version"
fi

echo "docker build . --no-cache -t $IMAGE_NAME:$new_version -f Dockerfile.$TYPE $PLATFORM"
docker build . -t $IMAGE_NAME:$new_version -f Dockerfile.$TYPE $PLATFORM

echo "docker tag $IMAGE_NAME:$new_version $REGISTRY/$IMAGE_NAME:$new_version"
docker tag $IMAGE_NAME:$new_version $REGISTRY/$IMAGE_NAME:$new_version

echo "docker push $REGISTRY/$IMAGE_NAME:$new_version"
docker push $REGISTRY/$IMAGE_NAME:$new_version

docker tag $IMAGE_NAME:$new_version $REGISTRY/$IMAGE_NAME:latest
docker push $REGISTRY/$IMAGE_NAME:latest
