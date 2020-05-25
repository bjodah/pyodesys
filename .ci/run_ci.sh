#!/bin/bash -xe

set -u
export PKG_NAME=$1
SUNDBASE=$2
set +u

if [ ! -e "$SUNDBASE/include/sundials/sundials_config.h" ]; then
    >&2 echo "Not a valid prefix for sundials: $SUNDBASE"
    exit 1
fi

mkdir -p $HOME/.config/pip/
echo -e "[global]\nno-cache-dir = false\ndownload-cache = $(pwd)/ci_cache/pip_cache" >$HOME/.config/pip/pip.conf
python3 -m pip install symcxx pysym  # unofficial backends, symengine is tested in the conda build

# (cd ./tmp/pycvodes;
CFLAGS="-isystem $SUNDBASE/include $CFLAGS" LDFLAGS="-Wl,--disable-new-dtags -Wl,-rpath,$SUNDBASE/lib -L$SUNDBASE/lib $LDFLAGS" python3 -m pip install pycvodes # setup.py install )
git clean -xfd # -e tmp/

# export CPATH=$SUNDBASE/include
# export LIBRARY_PATH=$SUNDBASE/lib
# export LD_LIBRARY_PATH=$SUNDBASE/lib

python3 setup.py sdist
PKG_VERSION=$(python3 setup.py --version)
(cd dist/; python3 -m pip install $PKG_NAME-$PKG_VERSION.tar.gz)
python3 -m pip install -e .[all]
python3 -m pytest -xv -k test_integrate_chained_robertson pyodesys/tests/test_robertson.py
export PYTHONHASHSEED=$(python3 -c "import random; print(random.randint(1,2**32-1))")
PYTHON="python3 -R" ./scripts/run_tests.sh --cov $PKG_NAME --cov-report html

./scripts/render_notebooks.sh
(cd $PKG_NAME/tests; jupyter nbconvert --log-level=INFO --to=html --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=600 *.ipynb)
./scripts/generate_docs.sh

# Test package without any 3rd party libraries that are in extras_require:
python3 -m pip install virtualenv
python3 -m virtualenv venv
git archive -o dist/$PKG_NAME-head.zip HEAD  # test pip installable zip (symlinks break)
set +u
(source ./venv/bin/activate; cd dist/; python3 -m pip install pytest $PKG_NAME-head.zip; python3 -m pytest --pyargs $PKG_NAME)
