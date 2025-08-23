#!/bin/bash

set -eux
export PKG_NAME=$1
SUNDBASE=$2
set +u

if [ ! -e "$SUNDBASE/include/sundials/sundials_config.h" ]; then
    >&2 echo "Not a valid prefix for sundials: $SUNDBASE"
    exit 1
fi
if [ -e /etc/profile.d/boost.sh ]; then
    source /etc/profile.d/boost.sh
    export CPATH=$BOOST_ROOT/include
fi
source $(compgen -G "/opt-3/cpython-v3.*-apt-deb/bin/activate")
python -m pip install --cache-dir $CI_WORKSPACE/cache-ci/pip_cache --upgrade-strategy=eager --upgrade cython "git+https://github.com/bjodah/pycompilation@master#egg=pycompilation"
# REPO_TEMP_DIR="$(mktemp -d)"
# trap 'rm -rf -- "$REPO_TEMP_DIR"' EXIT
# cp -ra . "$REPO_TEMP_DIR/."
# cd "$REPO_TEMP_DIR"

mkdir -p $HOME/.config/pip/
echo -e "[global]\nno-cache-dir = false\ndownload-cache = $CI_WORKSPACE/cache-ci/pip_cache" >$HOME/.config/pip/pip.conf
python -m pip install mako cython
python -m pip install --no-build-isolation "git+https://github.com/bjodah/symcxx#egg=symcxx" "git+https://github.com/bjodah/pysym#egg=pysym"  # unofficial backends, symengine is tested in the conda build

# (cd ./tmp/pycvodes;
SUND_CFLAGS="-isystem $SUNDBASE/include $CFLAGS"
SUND_LDFLAGS="-Wl,--disable-new-dtags -Wl,-rpath,$SUNDBASE/lib -L$SUNDBASE/lib $LDFLAGS"
CFLAGS="$SUND_CFLAGS $CXXFLAGS" CXXFLAGS="$SUND_CFLAGS $CXXFLAGS" LDFLAGS=$SUND_LDFLAGS python -m pip install --no-build-isolation pycvodes
CPLUS_INCLUDE_PATH="$BOOST_ROOT/include" python -m pip install --no-build-isolation "git+https://github.com/bjodah/pyodeint#egg=pyodeint"
python -m pip install --no-build-isolation "git+https://github.com/bjodah/pygslodeiv2#egg=pygslodeiv2"

python setup.py sdist
PKG_VERSION=$(python setup.py --version)
export PYODESYS_CVODE_FLAGS=$SUND_CFLAGS
export PYODESYS_CVODE_LDFLAGS=$SUND_LDFLAGS
(cd dist/; python -m pip install "$PKG_NAME-$PKG_VERSION.tar.gz[all]"; python -m pytest -v -x --pyargs $PKG_NAME)
python -m pip uninstall --yes $PKG_NAME
python -m pip install -e .[all]
python -m pytest -xv -k test_integrate_chained_robertson pyodesys/tests/test_robertson.py
export PYTHONHASHSEED=$(python3 -c "import random; print(random.randint(1,2**32-1))")
PYTHON="python -R" ./scripts/run_tests.sh --cov $PKG_NAME --cov-report html

( export PYTHONPATH=$(pwd); ./scripts/render_notebooks.sh; cd $PKG_NAME/tests; jupyter nbconvert --log-level=INFO --to=html --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=600 *.ipynb )
./scripts/generate_docs.sh

# Test package without any 3rd party libraries that are in extras_require:
python -m pip install virtualenv
python -m virtualenv venv
git archive -o dist/$PKG_NAME-head.zip HEAD  # test pip installable zip (symlinks break)
set +u
(source ./venv/bin/activate; cd dist/; python -m pip install pytest $PKG_NAME-head.zip; python -m pytest --pyargs $PKG_NAME)
