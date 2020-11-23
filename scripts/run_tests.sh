#!/bin/bash -e
# Usage
#   $ ./scripts/run_tests.sh
# or
#   $ ./scripts/run_tests.sh --cov pycvodes --cov-report html
${PYTHON:-python3} -m pytest -ra --doctest-modules --flakes -x --verbose $@
${PYTHON:-python3} -m doctest README.rst
rstcheck README.rst
