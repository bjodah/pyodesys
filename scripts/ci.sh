#!/bin/bash -xe

export PKG_NAME=$1

# Py3
conda create -q -n test3 python=3.5 scipy matplotlib sym sympy pysym symcxx pip pytest pytest-cov pytest-flakes pytest-pep8 pygslodeiv2 pyodeint pycvodes python-symengine appdirs pycompilation pycodeexport jupyter notebook
source activate test3
python setup.py install
# (cd /; python -m pytest --pyargs $1)

python -m pip install https://github.com/bjodah/pycvodes/archive/fixes.zip
python -m pip install https://github.com/bjodah/pygslodeiv2/archive/refactor.zip
python -m pip install https://github.com/bjodah/pyodeint/archive/refactor.zip

PYTHONPATH=$(pwd) ./scripts/run_tests.sh --cov $PKG_NAME --cov-report html
./scripts/coverage_badge.py htmlcov/ htmlcov/coverage.svg
#source deactivate

# Py2
conda create -q -n test2 python=2.7 scipy matplotlib sym sympy pysym symcxx pip pytest pytest-cov pygslodeiv2 pyodeint pycvodes python-symengine appdirs pycompilation pycodeexport
source activate test2
python setup.py sdist
(cd dist/; python -m pip install --force-reinstall --upgrade $PKG_NAME-*.tar.gz)

python -m pip install https://github.com/bjodah/pycvodes/archive/fixes.zip
python -m pip install https://github.com/bjodah/pygslodeiv2/archive/refactor.zip
python -m pip install https://github.com/bjodah/pyodeint/archive/refactor.zip

(cd /; python -m pytest --pyargs $PKG_NAME)

#source deactivate
source activate test3
python -m pip install .[all]
./scripts/render_notebooks.sh
./scripts/generate_docs.sh

! grep "DO-NOT-MERGE!" -R . --exclude ci.sh
