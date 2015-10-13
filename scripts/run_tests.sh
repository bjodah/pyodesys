#!/bin/bash -e
cd $(dirname $0)/..
python3 -m pytest --ignore build/ --ignore doc/ $@
python2 -m pytest --ignore build/ --ignore doc/ --doctest-modules --pep8 --flakes $@
python -m doctest README.rst
