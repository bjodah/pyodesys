#!/bin/bash -xe

export PKG_NAME=$1

for p in "${@:2}"
do
export CPATH=$p/include:$CPATH LIBRARY_PATH=$p/lib:$LIBRARY_PATH LD_LIBRARY_PATH=$p/lib:$LD_LIBRARY_PATH
done

python3 -m pip install symcxx pysym  # unofficial backends, symengine is tested in the conda build

python3 setup.py sdist
(cd dist/; python3 -m pip install $PKG_NAME-$(python3 ../setup.py --version).tar.gz)
python3 -m pip install --upgrade --upgrade-strategy only-if-needed .[all]

export PYTHONHASHSEED=$(python3 -c "import random; print(random.randint(1,2**32-1))")
PYTHON="python3 -R" ./scripts/run_tests.sh --cov $PKG_NAME --cov-report html

./scripts/render_notebooks.sh
(cd $PKG_NAME/tests; jupyter nbconvert --debug --to=html --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=300 *.ipynb)
./scripts/generate_docs.sh
