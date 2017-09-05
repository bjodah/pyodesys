#!/bin/bash -x
echo "Running tests..."
CPLUS_INCLUDE_PATH=$PREFIX/include LIBRARY_PATH=$PREFIX/lib MPLBACKEND=agg LLAPACK=openblas python -m pytest --verbose --pyargs pyodesys
