#!/bin/bash
#
# Usage:
#
#  $ ./scripts/host-jupyter-using-docker.sh
#  $ ./scripts/host-jupyter-using-docker.sh . 8888 ./scripts/environment
#
MOUNT=${1:-.}
PORT=${2:-8888}
DOCKERIMAGE=${3:-./scripts/environment}
if [[ "$MOUNT" == .* ]]; then
    MOUNT="$(pwd)/$MOUNT"
fi
if [[ "$DOCKERIMAGE" == ./* ]]; then
    DOCKERIMAGE=$(sudo docker build $DOCKERIMAGE | tee /dev/tty | tail -1 | cut -d' ' -f3)
fi
MYCMD="groupadd -f --gid \$HOST_GID \$HOST_WHOAMI; \
useradd --uid \$HOST_UID --gid \$HOST_GID --home /mount \$HOST_WHOAMI; \
sudo --preserve-env --login -u \$HOST_WHOAMI PYCVODES_LAPACK=lapack pip install --user -e .[all]; \
sudo --preserve-env --login -u \$HOST_WHOAMI LD_LIBRARY_PATH=/usr/local/lib /mount/.local/bin/jupyter notebook --no-browser --port $PORT --ip=* --notebook-dir examples"
docker run --rm --name "pyodesys_nb_$PORT" -p 127.0.0.1:$PORT:$PORT\
 -e HOST_WHOAMI=$(whoami) -e HOST_UID=$(id -u) -e HOST_GID=$(id -g)\
 -v $MOUNT:/mount -w /mount -it $DOCKERIMAGE /usr/bin/env bash -c "$MYCMD"
