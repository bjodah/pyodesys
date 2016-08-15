#!/bin/bash
cd examples/
source activate test2
ipython2 nbconvert --to=html --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=300 *.ipynb
../scripts/render_index.sh *.html
