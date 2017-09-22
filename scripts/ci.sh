#!/bin/bash -xe

export PKG_NAME=$1

for PY in python2 python3; do
    $PY -m pip install symcxx pysym  # unofficial backends, symengine is tested in the conda build
done

python3 setup.py sdist
(cd dist/; python3 -m pip install $PKG_NAME-$(python3 ../setup.py --version).tar.gz)


for PY in python2 python3; do
    $PY -m pip install --upgrade --upgrade-strategy only-if-needed .[all]
done

PYTHON=python2 ./scripts/run_tests.sh
PYTHON=python3 ./scripts/run_tests.sh --cov $PKG_NAME --cov-report html

./scripts/render_notebooks.sh
(cd $PKG_NAME/tests; OMP_NUM_THREADS=1 ANYODE_NUM_THREADS=1 jupyter nbconvert --debug --to=html --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=300 *.ipynb)
python3 -m pip install --user --force-reinstall docutils==0.12  # see https://github.com/sphinx-doc/sphinx/pull/3217
./scripts/generate_docs.sh


# Test package without any 3rd party libraries that are in extras_require:
python3 -m pip install virtualenv
python3 -m virtualenv venv
set +u
(source ./venv/bin/activate; python3 -m pip install pytest .; python3 -m pytest $PKG_NAME)
