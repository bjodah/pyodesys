#!/bin/bash

export PKG_NAME=$1

for p in "${@:2}"
do
export CPATH=$p/include:$CPATH LIBRARY_PATH=$p/lib:$LIBRARY_PATH LD_LIBRARY_PATH=$p/lib:$LD_LIBRARY_PATH
done


python2 -m pip install --user virtualenv
python2 -m virtualenv /tmp/test_py2
sed -i -E -e "/python_requires/d" setup.py
bash -c "source /tmp/test_py2/bin/activate; pip install pytest '.[all]' && pytest -rs --pyargs ${PKG_NAME}"

