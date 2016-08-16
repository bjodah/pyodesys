from __future__ import absolute_import

from ..symbolic import SymbolicSys
from .test_symbolic import decay_dydt_factory


def test_banded_jacobian():
    # Decay chain of 3 species (2 decays)
    # A --[k0=4]--> B --[k1=3]--> C
    k = [4, 3]
    odesys = SymbolicSys.from_callback(decay_dydt_factory(k), len(k)+1)
    bj = odesys.be.banded_jacobian(odesys.exprs, odesys.dep, 1, 0)
    assert bj.tolist() == [
        [-k[0], -k[1], 0],
        [k[0], k[1], 0],
    ]
