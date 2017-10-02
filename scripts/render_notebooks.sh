#!/bin/bash -e
cd examples/
for ipynb in *.ipynb; do
    if [[ $ipynb == "_native_standalone.ipynb" ]]; then
        continue  # issue with boost's program options
    fi
    jupyter nbconvert --debug --to=html --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=300 $ipynb
done
../scripts/render_index.sh *.html
