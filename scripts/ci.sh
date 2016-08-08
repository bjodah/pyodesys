#!/bin/bash -xeu

# Py3
conda create -q -n test3 python=3.5 scipy matplotlib sym sympy pysym symcxx pip pytest pytest-cov pytest-flakes pytest-pep8 pygslodeiv2 pyodeint pycvodes python-symengine
source activate test3
python setup.py install
# (cd /; python -m pytest --pyargs $1)
PYTHONPATH=$(pwd) ./scripts/run_tests.sh --cov $1 --cov-report html
./scripts/coverage_badge.py htmlcov/ htmlcov/coverage.svg
#source deactivate

# Py2
conda create -q -n test2 python=2.7 scipy matplotlib sym sympy pysym symcxx pip pytest pytest-cov pygslodeiv2 pyodeint pycvodes python-symengine
source activate test2
python setup.py sdist
pip install dist/*.tar.gz
(cd /; python -m pytest --pyargs $1)

! grep "DO-NOT-MERGE!" -R . --exclude ci.sh
