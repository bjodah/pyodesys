#!/bin/bash -u
#
# Usage:
#
#  $ ./get_sundials.sh 3.1.1 /opt/sun-3.1.1 -DLAPACK_ENABLE:BOOL=ON -DSUNDIALS_INDEX_TYPE:STRING="int32_t"
#  $ ./get_sundials.sh 2.7.0 /opt/sun-2.7.0 -DLAPACK_ENABLE:BOOL=OFF
#

function quiet_unless_fail {
    # suppresses function output unless exit status is != 0
    OUTPUT_FILE=$(tempfile)
    #/bin/rm --force /tmp/suppress.out 2>/dev/null
    EXECMD=${1+"$@"}
    $EXECMD > ${OUTPUT_FILE} 2>&1
    EXIT_CODE=$?
    if [ ${EXIT_CODE} -ne 0 ]; then
	cat ${OUTPUT_FILE}
	echo "The following command exited with exit status ${EXIT_CODE}: ${EXECMD}"
	/bin/rm ${OUTPUT_FILE}
	exit $?
    fi
    /bin/rm ${OUTPUT_FILE}
}

VERSION="$1"
PREFIX="$2"
if [ -d "$PREFIX" ]; then >&2 echo "Directory already exists: $PREFIX"; exit 1; fi
if [[ "$VERSION" == "2.7.0" ]]; then
    SUNDIALS_FNAME="sundials-2.7.0.tar.gz"
    SUNDIALS_MD5="c304631b9bc82877d7b0e9f4d4fd94d3"
    SUNDIALS_SHA256="d39fcac7175d701398e4eb209f7e92a5b30a78358d4a0c0fcc23db23c11ba104"
elif [[ "$VERSION" == "3.1.2" ]]; then
    SUNDIALS_FNAME="sundials-3.1.2.tar.gz"
    SUNDIALS_MD5="63304dafc935c94a0ad37832085384bc"
    SUNDIALS_SHA256="a8985bb1e851d90e24260450667b134bc13d71f5c6effc9e1d7183bd874fe116"
elif [[ "$VERSION" == "3.2.1" ]]; then
    SUNDIALS_FNAME="sundials-3.2.1.tar.gz"
    SUNDIALS_MD5="65c42e4fec7d1f4f4bcd670f9bbe31c0"
    SUNDIALS_SHA256="47d94d977ab2382cdcdd02f72a25ebd4ba8ca2634bbb2f191fe1636e71c86808"
elif [[ "$VERSION" == "4.0.2" ]]; then
    SUNDIALS_FNAME="sundials-4.0.2.tar.gz"
    SUNDIALS_MD5="2d840ed467ca491a3c1fe4ce67d2a99a"
    SUNDIALS_SHA256="6656d6938aed9142e61a001b1ed9f4ee4f7eaf003613bf5a887e98a85904d375"
elif [[ "$VERSION" == "4.1.0" ]]; then
    SUNDIALS_FNAME="sundials-4.1.0.tar.gz"
    SUNDIALS_MD5="f25bb0bc109ac3db0aaae13eadce559c"
    SUNDIALS_SHA256="280de1c27b2360170a6f46cb3799b2aee9dff3bddbafc8b08c291a47ab258aa5"
else
    >&2 echo "Unknown sundials version \"$VERSION\""
fi

SUNDIALS_URLS=(\
    "http://hera.physchem.kth.se/~repo/${SUNDIALS_MD5}/${SUNDIALS_FNAME}" \
    "http://davycrockett.mooo.com:49090/~repo/${SUNDIALS_SHA256}/${SUNDIALS_FNAME}" \
    "http://computation.llnl.gov/projects/sundials/download/${SUNDIALS_FNAME}" \
)
TIMEOUT=60  # 60 seconds

for URL in "${SUNDIALS_URLS[@]}"; do
    if echo $SUNDIALS_MD5 $SUNDIALS_FNAME | md5sum -c --; then
        echo "Found ${SUNDIALS_FNAME} with matching checksum, using that file."
    else
        echo "Downloading ${URL}..."
        timeout $TIMEOUT wget --quiet --tries=2 --timeout=$TIMEOUT $URL -O $SUNDIALS_FNAME || continue
    fi
    if echo $SUNDIALS_MD5 $SUNDIALS_FNAME | md5sum -c --; then
        tar xzf $SUNDIALS_FNAME
        mkdir sundials_build
        cd sundials_build
	( set -x; \
          cmake -DCMAKE_INSTALL_PREFIX:PATH="$PREFIX" \
		-DCMAKE_BUILD_TYPE:STRING="Release" \
		-DBUILD_SHARED_LIBS:BOOL=ON \
		-DBUILD_STATIC_LIBS:BOOL=OFF \
		-DEXAMPLES_ENABLE_C:BOOL=OFF \
		-DEXAMPLES_INSTALL:BOOL=OFF \
		-DOPENMP_ENABLE:BOOL=OFF \
		"${@:3}" "../sundials-$VERSION/"
	)
	if [[ $? -ne 0 ]]; then
	    >&2 echo "Cmake configuration failed."
	    exit 1
	fi
        quiet_unless_fail make VERBOSE=1 -j 1
        if [ $? -ne 0 ]; then
            >&2 echo "Building of sundials \"$VERSION\" failed."
            exit 1
        fi
        quiet_unless_fail make install
        if [ $? -ne 0 ]; then
            >&2 echo "Install of sundials \"$VERSION\" failed."
            exit 1
        fi
        cd ..
        rm -r sundials*
        exit 0
    fi
done
exit 1
