# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import math
from collections import OrderedDict
from ..util import import_
import numpy as np

from ..symbolic import SymbolicSys, TransformedSys, symmetricsys

sp = import_('sympy')

ys = [
    np.array([0.43976714474700634, 0.10031118340143896, 0.38147224769822524,
              1.7675704061619617e-11]),
    np.array([0.00064313123504933787, 0.00014677490343001067, 9.536739572030514e-05, 1.6877253332428752e-11])
]
ps = [
    [328.65, 39390, -135.3, 18010, 44960, 48.2,
     49320, -114.6, 1780, -47941.550570419757,
     107.24619394365152, 67486.458673807123, -170.63617364489184],
    [321.14999999999998, 39390, -135.30000000000001, 18010, 44960, 48.200000000000003,
     49320, -114.59999999999999, 1780, -34400.547966379738,
     -2.865040967667511, 93065.338440593958, 5.7581184659305222]
]


def _get_cetsa_odesys(molar_unitless, loglog, NativeSys=None, explicit_NL=False, MySys=None):
    # Never mind the details of this test case, it is from a real-world application.

    names = 'N U NL L A'.split()
    params = OrderedDict([
        ('T', 'T'),
        ('Ha_f', r'\Delta_f\ H^\neq'),
        ('Sa_f', r'\Delta_f S^\neq'),
        ('dCp_u', r'\Delta_u\ C_p'),
        ('He_u', r'\Delta_u H'),
        ('Tm_C', 'T_{m(C)}'),
        ('Ha_agg', r'\Delta_{agg}\ H^\neq'),
        ('Sa_agg', r'\Delta_{agg}\ S^\neq'),
        ('dCp_dis', r'\Delta_{dis}\ C_p'),
        ('Ha_as', r'\Delta_{as}\ H^\neq'),
        ('Sa_as', r'\Delta_{as}\ S^\neq'),
        ('He_dis', r'\Delta_{dis}\ H'),
        ('Se_dis', r'\Delta_{dis}\ S'),
    ])
    param_keys = list(params.keys())

    def Eyring(dH, dS, T, R, kB_over_h, be):
        return kB_over_h * T * be.exp(-(dH - T*dS)/(R*T))

    def get_rates(x, y, p, be=math, T0=298.15, T0C=273.15,
                  R=8.3144598,  # J K**-1 mol**-1,  J = Nm, but we keep activation energies in Joule)
                  kB_over_h=1.38064852e-23 / 6.62607004e-34):  # K**-1 s**-1
        pd = dict(zip(param_keys, p))
        He_u_T = pd['He_u'] + pd['dCp_u'] * (pd['T'] - T0)
        He_dis_T = pd['He_dis'] + pd['dCp_dis'] * (pd['T'] - T0)
        Se_u = pd['He_u']/(T0C + pd['Tm_C']) + pd['dCp_u']*be.log(pd['T']/T0)
        Se_dis = pd['Se_dis'] + pd['dCp_dis']*be.log(pd['T']/T0)

        def C(k):
            return y[names.index(k)]

        return {
            'unfold': C('N')*Eyring(He_u_T + pd['Ha_f'], pd['Sa_f'] + Se_u, pd['T'], R, kB_over_h, be),
            'fold': C('U')*Eyring(pd['Ha_f'], pd['Sa_f'], pd['T'], R, kB_over_h, be),
            'aggregate': C('U')*Eyring(pd['Ha_agg'], pd['Sa_agg'], pd['T'], R, kB_over_h, be),
            'dissociate': C('NL')*Eyring(He_dis_T + pd['Ha_as'], Se_dis + pd['Sa_as'], pd['T'], R, kB_over_h, be),
            'associate': C('N')*C('L')*Eyring(pd['Ha_as'], pd['Sa_as'], pd['T'], R, kB_over_h, be) / molar_unitless
        }

    def f(x, y, p, be=math):
        r = get_rates(x, y, p, be)
        dydx = {
            'N': r['fold'] - r['unfold'] + r['dissociate'] - r['associate'],
            'U': r['unfold'] - r['fold'] - r['aggregate'],
            'A': r['aggregate'],
            'L': r['dissociate'] - r['associate'],
            'NL': r['associate'] - r['dissociate']
        }
        return [dydx[k] for k in (names if explicit_NL else names[:-1])]

    if loglog:
        logexp = sp.log, sp.exp
        if NativeSys:
            class SuperClass(TransformedSys, NativeSys):
                pass
        else:
            SuperClass = TransformedSys
        MySys = symmetricsys(
            logexp, logexp, SuperClass=SuperClass, exprs_process_cb=lambda exprs: [
                sp.powsimp(expr.expand(), force=True) for expr in exprs])
    else:
        MySys = NativeSys or SymbolicSys

    return MySys.from_callback(f, len(names) - (0 if explicit_NL else 1), len(param_keys),
                               names=names if explicit_NL else names[:-1])
