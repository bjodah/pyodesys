#!/bin/bash -xeu
# Usage:
#
#    $ ./scripts/release.sh v1.2.3 ~/anaconda2/bin
#

if [[ $1 != v* ]]; then
    echo "Argument does not start with 'v'"
    exit 1
fi
VERSION=${1#v}
./scripts/check_clean_repo_on_master.sh
cd $(dirname $0)/..
# PKG will be name of the directory one level up containing "__init__.py" 
PKG=$(find . -maxdepth 2 -name __init__.py -print0 | xargs -0 -n1 dirname | xargs basename)
! grep --include "*.py" "will_be_missing_in='$VERSION'" -R $PKG/  # see deprecation()
PKG_UPPER=$(echo $PKG | tr '[:lower:]' '[:upper:]')
./scripts/run_tests.sh
env ${PKG_UPPER}_RELEASE_VERSION=v$VERSION python setup.py sdist
env ${PKG_UPPER}_RELEASE_VERSION=v$VERSION ./scripts/generate_docs.sh
for CONDA_PY in 2.7 3.4 3.5; do
    PATH=$2:$PATH ./scripts/build_conda_recipe.sh v$VERSION --python $CONDA_PY
done

# All went well, add a tag and push it.
git tag -a v$VERSION -m v$VERSION
git push
git push --tags
twine upload dist/${PKG}-$VERSION.tar.gz

echo "Make a release on github with the new tag and manually attach the new .tar.gz file from ./dist/"
echo "Then run ./scripts/post_release.sh $1 $2 MYGITHUBUSERNAME"
