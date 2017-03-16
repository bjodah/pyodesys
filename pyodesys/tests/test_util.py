from __future__ import absolute_import

import pytest

from ..symbolic import SymbolicSys
from ..util import requires, import_
from .test_symbolic import decay_dydt_factory


@requires('sym', 'scipy')
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


@requires('numpy')
def test_import_():
    sqrt, sin = import_('numpy', 'sqrt', 'sin')
    assert sqrt(4) == 2 and sin(0) == 0

    foo, bar = import_('numpy', 'foo', 'bar')
    with pytest.raises(AttributeError):
        foo.baz
    with pytest.raises(AttributeError):
        bar(3)

    qux = import_('qux')
    with pytest.raises(ImportError):
        qux.__name__
