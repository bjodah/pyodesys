#!/bin/bash
if grep "DO-NOT-MERGE!" -R . --exclude $(basename $BASH_SOURCE); then
    exit 1
fi
