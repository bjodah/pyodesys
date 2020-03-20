#!/bin/bash -e
QUIET_EXIT_CODE=0
function quiet_unless_fail {
    # suppresses function output unless exit status is != 0
    OUTPUT_FILE=$(tempfile)
    #/bin/rm --force /tmp/suppress.out 2>/dev/null
    EXECMD=${1+"$@"}
    $EXECMD > ${OUTPUT_FILE} 2>&1
    QUIET_EXIT_CODE=$?
    if [ ${QUIET_EXIT_CODE} -ne 0 ]; then
	cat ${OUTPUT_FILE}
	echo "The following command exited with exit status ${QUIET_EXIT_CODE}: ${EXECMD}"
    fi
    /bin/rm ${OUTPUT_FILE}
}


cd examples/

PREC=`python3 -c "from pycvodes._config import env; print(env.get('SUNDIALS_PRECISION', 'double'))"`
set -x
for ipynb in *.ipynb; do
    if [[ $ipynb == "_native_standalone.ipynb" ]]; then
        continue  # issue with boost's program options
    fi
    if [[ $PREC != "double" && $ipynb == "_robertson.ipynb" ]]; then
        continue
    fi
    if [[ $ipynb == "_bench_native_odesys_multi.ipynb" ]]; then
        continue
    fi
    #quiet_unless_fail
    jupyter nbconvert --debug --to=html --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=900 "${ipynb}" \
        | grep -v -e "^\[NbConvertApp\] content: {'data':.*'image/png'"
    #if [ ${QUIET_EXIT_CODE} -ne 0 ]; then
    #    exit ${QUIET_EXIT_CODE}
    #fi
done
../scripts/render_index.sh *.html
