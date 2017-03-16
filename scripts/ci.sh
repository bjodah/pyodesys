#!/bin/bash -xe

export PKG_NAME=$1

echo "deb http://ppa.launchpad.net/symengine/ppa/ubuntu xenial main" >>/etc/apt/sources.list
apt-get update
apt-get install --quiet --assume-yes --no-install-recommends python-symengine python3-symengine

for PY in python2 python3; do
    $PY -c "import symengine"  # make sure symengine is installed
    $PY -m pip install symcxx pysym  # unofficial backends
    $PY -m pip install git+https://github.com/bjodah/sympy@expm1-log1p  # DO-NOT-MERGE!
done

python setup.py sdist
(cd dist/; python -m pip install --force-reinstall $PKG_NAME-*.tar.gz)


for PY in python2 python3; do
    $PY -m pip install --upgrade .[all]
done

PYTHON=python2 ./scripts/run_tests.sh
PYTHON=python3 ./scripts/run_tests.sh --cov $PKG_NAME --cov-report html

./scripts/render_notebooks.sh
(cd $PKG_NAME/tests; jupyter nbconvert --debug --to=html --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=300 *.ipynb)
python3 -m pip install --user --force-reinstall docutils==0.12  # see https://github.com/sphinx-doc/sphinx/pull/3217
./scripts/generate_docs.sh


# Test package without any 3rd party libraries that are in extras_require:
python3 -m pip install virtualenv
python3 -m virtualenv venv
set +u
(source ./venv/bin/activate; python3 -m pip install pytest .; python3 -m pytest $PKG_NAME)
