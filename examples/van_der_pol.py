#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sympy as sp
import numpy as np

from pyodesys import SymbolicSys


def main(y0='1,0', mu=1.0, tend=10., nt=2, savefig='None', plot=False,
         savetxt='None', solver='scipy', dpi=100, kwargs=''):
    assert nt > 1
    y = sp.symarray('y', 2)
    f = [y[1], -y[0] + mu*y[1]*(1 - y[0]**2)]
    odesys = SymbolicSys(zip(y, f))
    tout = np.linspace(0, tend, nt)
    y0 = map(float, y0.split(','))
    kwargs = dict(eval(kwargs) if kwargs else {})
    out = odesys.integrate(solver, tout, y0, **kwargs)
    if savetxt != 'None':
        np.savetxt(out, savetxt)
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(out[:, 0], out[:, 1:])
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
