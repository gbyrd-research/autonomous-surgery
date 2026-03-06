#!/usr/bin/env bash

set -e

INTERACTIVE=false

# Parse CLI args
for arg in "$@"; do
    case $arg in
        -i|--interactive)
            INTERACTIVE=true
            shift
            ;;
        *)
            ;;
    esac
done

# # ensure ssh agent is initialized
# eval "$(ssh-agent -s)"
# ssh-add ~/.ssh/id_rsa

# run the container in detached mode
docker compose up -d

# enter the container in interactive mode (only if requested)
if [ "$INTERACTIVE" = true ]; then
    docker exec -it autonomous_surgery_dev /bin/bash
else
    echo "Container started in detached mode."
    echo "Use --interactive (or -i) to enter the container."
fi