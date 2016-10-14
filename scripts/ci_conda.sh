#!/bin/bash -xe

export PKG_NAME=$1

conda config --add channels symengine/label/dev
conda config --add channels bjodah


# Py3
conda create -q -n test3 python=3.5 scipy matplotlib sym sympy pysym symcxx pip pytest pytest-cov pytest-flakes pytest-pep8 pygslodeiv2 pyodeint pycvodes python-symengine appdirs pycompilation pycodeexport jupyter notebook
source activate test3
python setup.py install
# (cd /; python -m pytest --pyargs $1)

python -m pip install https://github.com/bjodah/pygslodeiv2/releases/download/v0.6.0/pygslodeiv2-0.6.0.tar.gz
python -m pip install https://github.com/bjodah/pyodeint/releases/download/v0.7.0/pyodeint-0.7.0.tar.gz
python -m pip install https://github.com/bjodah/pycvodes/releases/download/v0.6.0/pycvodes-0.6.0.tar.gz

PYTHONPATH=$(pwd) ./scripts/run_tests.sh --cov $PKG_NAME --cov-report html
./scripts/coverage_badge.py htmlcov/ htmlcov/coverage.svg
#source deactivate

# Py2
conda create -q -n test2 python=2.7 scipy matplotlib sym sympy pysym symcxx pip pytest pytest-cov pygslodeiv2 pyodeint pycvodes python-symengine appdirs pycompilation pycodeexport
source activate test2
python setup.py sdist
(cd dist/; python -m pip install --force-reinstall --upgrade $PKG_NAME-*.tar.gz)

python -m pip install https://github.com/bjodah/pygslodeiv2/releases/download/v0.6.0/pygslodeiv2-0.6.0.tar.gz
python -m pip install https://github.com/bjodah/pyodeint/releases/download/v0.7.0/pyodeint-0.7.0.tar.gz
python -m pip install https://github.com/bjodah/pycvodes/releases/download/v0.6.0/pycvodes-0.6.0.tar.gz

(cd /; python -m pytest --pyargs $PKG_NAME)

#source deactivate
source activate test3
python -m pip install .[all]
./scripts/render_notebooks.sh
./scripts/generate_docs.sh
