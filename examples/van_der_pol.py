#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sympy as sp
import numpy as np

from pyodesys.symbolic import SymbolicSys
from pyodesys.util import stack_1d_on_left


def main(y0='1,0', mu=1.0, tend=10., nt=2, savefig='None', plot=False,
         savetxt='None', solver='scipy', dpi=100, kwargs='', verbose=False):
    assert nt > 1
    y = sp.symarray('y', 2)
    p = sp.Symbol('p', real=True)
    f = [y[1], -y[0] + p*y[1]*(1 - y[0]**2)]
    odesys = SymbolicSys(zip(y, f), params=[p], names=True)
    tout = np.linspace(0, tend, nt)
    y0 = list(map(float, y0.split(',')))
    kwargs = dict(eval(kwargs) if kwargs else {})
    xout, yout, info = odesys.integrate(solver, tout, y0, [mu], **kwargs)
    if verbose:
        print(info)
    if savetxt != 'None':
        np.savetxt(stack_1d_on_left(xout, yout), savetxt)
    if plot:
        import matplotlib.pyplot as plt
        odesys.plot_result(xout, yout)
        plt.legend()
        if savefig != 'None':
            plt.savefig(savefig, dpi=dpi)
        else:
            plt.show()

if __name__ == '__main__':
    try:
        import argh
        argh.dispatch_command(main)
    except ImportError:
        import sys
        if len(sys.argv) > 1:
            import warnings
            warnings.warn("Ignoring parameters run "
                          "'pip install --user argh' to fix.")
        main()
