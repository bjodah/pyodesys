#!/bin/bash -ue
#
# Usage:
#
#  $ ./scripts/host-jupyter-using-docker.sh
#  $ ./scripts/host-jupyter-using-docker.sh . 8888 ./scripts/environment
#  $ ./scripts/host-jupyter-using-docker.sh . 0 ./scripts/environment
#
# Arguments: mount-path, port-number, Dockerfile-path
#
# If port == 0: the test suite is run
#
MOUNT=${1:-.}
PORT=${2:-8888}
DOCKERIMAGE=${3:-./scripts/environment}
RUNTESTS=${4:-"0"}
if [[ "$MOUNT" == .* ]]; then
    MOUNT="$(pwd)/$MOUNT"
fi
if [[ "$DOCKERIMAGE" == ./* ]]; then
    DOCKERIMAGE=$(sudo docker build $DOCKERIMAGE | tee /dev/tty | tail -1 | cut -d' ' -f3)
fi
if [[ "$PORT" == "0" ]]; then
    LOCALCMD="pytest -rs --pyargs pyodesys --slow --veryslow"
    PORTFWD=""
else
    LOCALCMD="jupyter notebook --no-browser --port $PORT --ip=* --notebook-dir examples"
    PORTFWD="-p 127.0.0.1:$PORT:$PORT"
fi
MYCMD="groupadd -f --gid \$HOST_GID \$HOST_WHOAMI; \
useradd --uid \$HOST_UID --gid \$HOST_GID --home /mount \$HOST_WHOAMI; \
sudo --preserve-env --login -u \$HOST_WHOAMI pip install symcxx quantities; \
sudo --preserve-env --login -u \$HOST_WHOAMI PYCVODES_LAPACK=lapack pip install --user -e .[all]; \
sudo --preserve-env --login -u \$HOST_WHOAMI LD_LIBRARY_PATH=/usr/local/lib MPLBACKEND=Agg /mount/.local/bin/$LOCALCMD"
docker run --rm --name "pyodesys_nb_$PORT" $PORTFWD \
 -e HOST_WHOAMI=$(whoami) -e HOST_UID=$(id -u) -e HOST_GID=$(id -g)\
 -v $MOUNT:/mount -w /mount -it $DOCKERIMAGE /usr/bin/env bash -c "$MYCMD"
