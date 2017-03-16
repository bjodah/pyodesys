# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import codecs
import gzip
import pickle
from functools import reduce
from itertools import chain
from operator import mul

from pyodesys.symbolic import SymbolicSys


def get_odesys(reac, prod, names, SymbSys, inact_reac=None, **kwargs):
    nr = len(reac)
    ns = len(names)
    if inact_reac is None:
        inact_reac = [[0]*ns]*nr
    if nr != len(prod) or nr != len(inact_reac):
        raise ValueError("reac, prod & inact_reac must be of same length")

    if any(len(s) != ns for s in chain(reac, prod, inact_reac)):
        raise ValueError("Inconsistent lengths of arrays")

    def reaction_rates(t, C, params):
        for i, k in enumerate(params):
            yield reduce(mul, [k] + [C[j]**n for j, n in enumerate(reac[i])])

    def dCdt(t, C, params):
        if kwargs.get('dep_by_name') is True:
            C = [C[k] for k in names]

        rates = list(reaction_rates(t, C, params))
        result = [0]*ns
        for r, sr, sp, si in zip(rates, reac, prod, inact_reac):
            for i in range(ns):
                result[i] += r*(sp[i] - sr[i] - si[i])

        if kwargs.get('dep_by_name') is True:
            result = {k: v for k, v in zip(names, result)}
        return result

    return SymbSys.from_callback(dCdt, ns, nr, names=names, **kwargs)


def from_b64_gz_pkl(data, pop=('params',), **kwargs):
    imported = pickle.loads(gzip.decompress(codecs.decode(data.encode('utf-8'), 'base64')))
    popped = [imported.pop(k) for k in pop]
    if 'SymbSys' not in kwargs:
        kwargs['SymbSys'] = SymbolicSys
    if 'dep_by_name' not in kwargs:
        kwargs['dep_by_name'] = True
    imported.update(kwargs)
    return get_odesys(**imported), popped


radiolysis1 = """
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
"""
