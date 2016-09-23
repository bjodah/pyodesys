#!/bin/bash
cd examples/
python -m pip install .[all]
jupyter nbconvert --to=html --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=300 *.ipynb
../scripts/render_index.sh *.html
