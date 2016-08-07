from __future__ import (absolute_import, division, print_function)

from ..symbolic import SymbolicSys
from .test_core import vdp_f

import numpy as np
import pytest


def _get_symbolic_system(use_pysym):
    return SymbolicSys.from_callback(vdp_f, 2, 1, backend='pysym' if use_pysym else None)


@pytest.mark.parametrize('use_pysym', [True, False])
def test_pysym_SymbolicSys_from_callback(use_pysym):
    ss = _get_symbolic_system(use_pysym)
    xout, yout, info = ss.integrate([0, 1, 2], [1, 0], params=[2.0])
    # blessed values:
    ref = [[1, 0], [0.44449086, -1.32847148], [-1.89021896, -0.71633577]]
    assert np.allclose(yout, ref)
    assert info['nfev'] > 0
