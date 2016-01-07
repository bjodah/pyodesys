#!/bin/bash -eux
#
# Careful - this script rebases and pushes forcefully!
#


UPLOAD_DIR=$1
GITHUB_REPO=$2
OVERWRITE_UPLOAD_BRANCH=$3
OUTPUTDIR=$4
git config --global user.name ${5:-drone}
git config --global user.email ${6:-"drone@nohost.com"}
WORKDIR=$(pwd)
git clone --quiet git://github.com/${GITHUB_REPO} $OUTPUTDIR > /dev/null
cd $OUTPUTDIR
git checkout --orphan $OVERWRITE_UPLOAD_BRANCH
git rm -rf . > /dev/null
cd $WORKDIR
cp -r ${UPLOAD_DIR}/. $OUTPUTDIR/
cd $OUTPUTDIR
git add -f . > /dev/null
git commit -m "Lastest docs from successful drone build (hash: ${DRONE_COMMIT})"
#git push -f origin $OVERWRITE_UPLOAD_BRANCH
