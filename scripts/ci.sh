#!/bin/bash -xe

export PKG_NAME=$1

echo "deb http://ppa.launchpad.net/symengine/ppa/ubuntu xenial main" >>/etc/apt/sources.list
apt-get update
apt-get install python-symengine python3-symengine

for PY in python2 python3; do
    $PY -c "import symengine"  # make sure symengine is installed
    $PY -m pip install symcxx pysym  # unofficial backends
done

python setup.py sdist
(cd dist/; python -m pip install --force-reinstall --upgrade $PKG_NAME-*.tar.gz)


for PY in python2 python3; do
    $PY -m pip install --upgrade pip
    $PY -m pip install --upgrade .[all]
done

PYTHON=python2 ./scripts/run_tests.sh
PYTHON=python3 ./scripts/run_tests.sh --cov $PKG_NAME --cov-report html

./scripts/render_notebooks.sh
./scripts/generate_docs.sh
