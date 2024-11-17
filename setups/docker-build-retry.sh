#!/bin/bash

# Default values
DOCKERFILE=""
TAG=""
ROOT="."
MAX_RETRIES=30

# Parse command line arguments
while getopts "f:t:" opt; do
    case $opt in
        f) DOCKERFILE="$OPTARG";;
        t) TAG="$OPTARG";;
        *) echo "Usage: $0 -f <dockerfile> -t <tag> [root_dir]"; exit 1;;
    esac
done

# Shift past the options
shift $((OPTIND-1))

# Get root directory if provided
if [ $# -eq 1 ]; then
    ROOT="$1"
fi

# Validate required parameters
if [ -z "$DOCKERFILE" ] || [ -z "$TAG" ]; then
    echo "Usage: $0 -f <dockerfile> -t <tag> [root_dir]"
    exit 1
fi

attempt=1

while [ $attempt -le $MAX_RETRIES ]; do
    echo "Attempt $attempt of $MAX_RETRIES"
    echo "Running: docker build -f $DOCKERFILE -t $TAG $ROOT"
    
    # Run docker build and capture its exit status while showing output in real-time
    output_file=$(mktemp)
    docker build -f "$DOCKERFILE" -t "$TAG" "$ROOT" 2>&1 | tee "$output_file"
    build_status=${PIPESTATUS[0]}  # Capture the exit status of docker build, not tee
    
    if [ $build_status -eq 0 ]; then
        echo "Build successful on attempt $attempt"
        rm "$output_file"
        exit 0
    fi
    
    # Check if error contains "Bad Request"
    if grep -i "400.*Bad Request.*IP:" "$output_file" > /dev/null; then
        echo "Bad Request error detected, retrying..."
        attempt=$((attempt + 1))
        sleep 5  # Add a small delay between retries
    else
        echo "Failed with non-Bad Request error"
        rm "$output_file"
        exit 1
    fi
    
    rm "$output_file"
done

echo "Maximum retries ($MAX_RETRIES) reached. Build failed."
exit 1