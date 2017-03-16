#!/bin/bash
if grep "DO-NOT-MERGE!" -R . --exclude $0; then
    exit 1
fi
