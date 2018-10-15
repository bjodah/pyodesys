#!/bin/bash -e
cd examples/

PREC=`python3 -c "from pycvodes._config import env; print(env.get('SUNDIALS_PRECISION', 'double'))"`

for ipynb in *.ipynb; do
    if [[ $ipynb == "_native_standalone.ipynb" ]]; then
        continue  # issue with boost's program options
    fi
    if [[ $PREC != "double" && $ipynb == "_robertson.ipynb" ]]; then
        continue
    fi
    jupyter nbconvert --debug --to=html --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=420 $ipynb
done
../scripts/render_index.sh *.html
