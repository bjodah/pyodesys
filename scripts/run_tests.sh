#!/bin/bash
cd $(dirname $0)/..
python2.7 -m pytest --ignore build/ --ignore doc/ $@
python3 -m pytest --ignore build/ --ignore doc/ --doctest-modules --pep8 --flakes $@
python3 -m doctest README.rst
