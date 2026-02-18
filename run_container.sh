# ensure ssh agent is initialized
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa

# run the container in detached mode
docker compose up -d

# enter the container in interactive mode
docker exec -it autonomous_surgery_dev /bin/bash