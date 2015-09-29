#!/bin/bash -e
PKG_NAME=${1}
if [[ ! -z $PKG_NAME ]]; then
    COV_FLAGS="--cov $PKG_NAME --cov-report html"
else
    COV_FLAGS=""
fi
cd $(dirname $0)/..
python2 -m pytest --ignore build/ --ignore doc/ --doctest-modules --pep8 --flakes $COV_FLAGS
python3 -m pytest --ignore build/ --ignore doc/
python -m doctest README.rst
