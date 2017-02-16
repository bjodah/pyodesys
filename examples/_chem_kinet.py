# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

from functools import reduce
from itertools import chain
from operator import mul


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
