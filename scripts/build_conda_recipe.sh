#!/bin/bash -ex
# Usage:
#
#    $ ./scripts/build_conda_recipe.sh v1.2.3
#
if [ ! -z $1 ]; then
    echo $1>__conda_version__.txt
    trap "rm __conda_version__.txt" EXIT SIGINT SIGTERM
fi
conda build conda-recipe
