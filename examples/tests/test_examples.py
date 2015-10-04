# -*- coding: utf-8 -*-

import glob
import os
import subprocess
import sys

import pytest


tests = glob.glob(os.path.join(os.path.dirname(__file__), '../*.py'))


@pytest.mark.parametrize('pypath', tests)
def test_examples(pypath):
    py_exe = 'python3' if sys.version_info.major == 3 else 'python'
    p = subprocess.Popen([py_exe, pypath])
    assert p.wait() == 0  # SUCCESS==0

    py_exe = 'python3' if sys.version_info.major == 3 else 'python'
    p = subprocess.Popen([py_exe, pypath])
    assert p.wait() == 0  # SUCCESS==0
