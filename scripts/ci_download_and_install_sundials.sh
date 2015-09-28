#!/bin/bash
BUILD_TYPE=${1:-Debug}
TIMEOUT=${2:-60}
SUNDIALS_FNAME="sundials-2.6.2.tar.gz"
SUNDIALS_MD5="3deeb0ede9f514184c6bd83ecab77d95"
SUNDIALS_URLS=(\
"http://hera.physchem.kth.se/~bjorn/${SUNDIALS_FNAME}" \
"http://pkgs.fedoraproject.org/repo/pkgs/sundials/${SUNDIALS_FNAME}/${SUNDIALS_MD5}/${SUNDIALS_FNAME}" \
"http://computation.llnl.gov/casc/sundials/download/code/${SUNDIALS_FNAME}" \
)
tmpdir=$(mktemp -d)
trap "rm -r $tmpdir" EXIT SIGINT SIGTERM
cd $tmpdir
for URL in "${SUNDIALS_URLS[@]}"; do
    echo "Downloading ${URL}..."
    timeout $TIMEOUT wget --quiet --tries=2 --timeout=$TIMEOUT $URL
    if [ $? -eq 0 ]; then
        echo $SUNDIALS_MD5 $SUNDIALS_FNAME | md5sum -c --
        if [ $? -eq 0 ]; then
            tar xzf $SUNDIALS_FNAME
            mkdir sundials_build
            cd sundials_build
            cmake -DCMAKE_BUILD_TYPE:STRING="$BUILD_TYPE" -DBUILD_SHARED_LIBS:BOOL="1" -DBUILD_STATIC_LIBS:BOOL="0" -DEXAMPLES_ENABLE:BOOL="0" \
                -DEXAMPLES_INSTALL:BOOL="0" -DLAPACK_ENABLE:BOOL="1" -DOPENMP_ENABLE:BOOL="0" ../sundials-*/
            if [[ $? != 0 ]]; then
                echo "cmake of sundials failed."
                exit 1
            fi
            make
            if [[ $? != 0 ]]; then
                echo "make of sundials failed."
                exit 1
            fi
            sudo make install
            if [[ $? != 0 ]]; then
                echo "make install of sundials failed."
                exit 1
            fi
            exit 0
        else
            rm ${SUNDIALS_FNAME}
        fi
    fi    
done
exit 1
