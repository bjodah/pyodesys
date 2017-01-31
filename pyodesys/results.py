# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import numpy as np

from .plotting import plot_result, plot_phase_plane, info_vlines
from .util import import_

CubicSpline = import_('scipy.interpolate', 'CubicSpline')


class Result(object):

    def __init__(self, xout, yout, params, info, odesys):
        self.xout = xout
        self.yout = yout
        self.params = params
        self.info = info
        self.odesys = odesys

    def __len__(self):
        return 3

    def __getitem__(self, key):
        if key == 0:
            return self.xout
        elif key == 1:
            return self.yout
        elif key == 2:
            return self.info
        elif key == 3:
            raise StopIteration
        else:
            raise KeyError("Invalid key: %s (for backward compatibility reasons)." % str(key))

    def between(self, lower, upper, xdata=None, ydata=None):
        """ Get results inside span for independent variable """
        if xdata is None:
            xdata = self.xout
        if ydata is None:
            ydata = self.yout
        select_u = xdata < upper
        xtmp, ytmp = xdata[..., select_u], ydata[..., select_u, :]
        select_l = xtmp > lower
        return xtmp[..., select_l], ytmp[..., select_l, :]

    def at(self, x, use_deriv=False, xdata=None, ydata=None):
        """ Returns interpolated result at a given time and an interpolation error-estimate """
        if xdata is None:
            xdata = self.xout
        if ydata is None:
            ydata = self.yout

        if x == xdata[0]:
            res = ydata[0, :]
            err = res*0
        elif x == xdata[-1]:
            res = ydata[-1, :]
            err = res*0
        else:
            idx = np.argmax(xdata > x)
            if idx == 0:
                raise ValueError("x outside bounds")
            idx_l = max(0, idx - 2)
            idx_u = min(xdata.size, idx_l + 4)
            slc = slice(idx_l, idx_u)
            res_cub = CubicSpline(xdata[slc], ydata[slc, :])(x)
            x0, x1 = xdata[idx - 1], xdata[idx]
            y0, y1 = ydata[idx - 1, :], ydata[idx, :]
            xspan, yspan = x1 - x0, y1 - y0
            avgx, avgy = .5*(x0 + x1), .5*(y0 + y1)
            if use_deriv:
                # y = a + b*x + c*x**2 + d*x**3
                # dydx = b + 2*c*x + 3*d*x**2
                y0p, y1p = [np.asarray(self.odesys.f_cb(x, y, self.params))*xspan for y in (y0, y1)]
                lsx = (x - x0)/xspan
                d = y0p + y1p + 2*y0 - 2*y1
                c = -2*y0p - y1p - 3*y0 + 3*y1
                b, a = y0p, y0
                res_poly = a + b*lsx + c*lsx**2 + d*lsx**3
                res, err = res_poly, np.abs(res_poly - res_cub)
            else:
                res_lin = avgy + yspan/xspan*(x - avgx)
                res, err = res_cub, np.abs(res_cub - res_lin)

        return res, err

    def _internal(self, key, override=None):
        if override is None:
            return self.info['internal_' + key]
        else:
            return override

    def stiffness(self, xyp=None, eigenvals_cb=None):
        """ Running stiffness ratio from last integration.

        Calculate sittness ratio, i.e. the ratio between the largest and
        smallest absolute eigenvalue of the jacobian matrix. The user may
        supply their own routine for calculating the eigenvalues, or they
        will be calculated from the SVD (singular value decomposition).
        Note that calculating the SVD for any but the smallest Jacobians may
        prove to be prohibitively expensive.

        Parameters
        ----------
        xyp : length 3 tuple (default: None)
            internal_xout, internal_yout, internal_params, taken
            from last integration if not specified.
        eigenvals_cb : callback (optional)
            Signature (x, y, p) (internal variables), when not provided an
            internal routine will use ``self.j_cb`` and ``scipy.linalg.svd``.

        """
        if eigenvals_cb is None:
            if self.odesys.band is not None:
                raise NotImplementedError
            eigenvals_cb = self.odesys._jac_eigenvals_svd

        if xyp is None:
            x, y, intern_p = self._internal('xout'), self._internal('yout'), self._internal('params')
        else:
            x, y, intern_p = self.pre_process(*xyp)

        singular_values = []
        for xval, yvals in zip(x, y):
            singular_values.append(eigenvals_cb(xval, yvals, intern_p))

        return (np.abs(singular_values).max(axis=-1) /
                np.abs(singular_values).min(axis=-1))

    def _plot(self, cb, internal_xout=None, internal_yout=None, internal_params=None, **kwargs):
        kwargs = kwargs.copy()
        if 'x' in kwargs or 'y' in kwargs or 'params' in kwargs:
            raise ValueError("x and y from internal_xout and internal_yout")

        if 'post_processors' not in kwargs:
            kwargs['post_processors'] = self.odesys.post_processors

        if 'names' in kwargs:
            if 'indices' not in kwargs and getattr(self.odesys, 'names', None) is not None:
                kwargs['indices'] = [self.odesys.names.index(n) for n in kwargs['names']]
                kwargs['names'] = self.odesys.names
        else:
            kwargs['names'] = getattr(self.odesys, 'names', None)

        if 'latex_names' not in kwargs:
            _latex_names = getattr(self.odesys, 'latex_names', None)
            if _latex_names is not None and not all(ln is None for ln in _latex_names):
                kwargs['latex_names'] = _latex_names

        return cb(self._internal('xout', internal_xout),
                  self._internal('yout', internal_yout),
                  self._internal('params', internal_params), **kwargs)

    def plot(self, info_vlines_kw=None, between=None, **kwargs):
        """ Plots the integrated dependent variables from last integration.

        Parameters
        ----------
        info_vlines_kw : dict
            Keyword arguments passed to :func:`.plotting.info_vlines`,
            an empty dict will be used if `True`. Need to pass `ax` when given.
        indices : iterable of int
        between : length 2 tuple
        names : iterable of str
        \*\*kwargs:
            See :func:`pyodesys.plotting.plot_result`
        """
        if between is not None:
            if 'internal_xout' in kwargs or 'internal_yout' in kwargs:
                raise ValueError("internal_xout/internal_yout & between given.")
            kwargs['internal_xout'], kwargs['internal_yout'] = self.between(
                *between, xdata=self._internal('xout'), ydata=self._internal('yout'))
        if info_vlines_kw is not None:
            if info_vlines_kw is True:
                info_vlines_kw = {}
            info_vlines(kwargs['ax'], self.xout, self.info, **info_vlines_kw)
            self._plot(plot_result, plot_kwargs_cb=lambda *args, **kwargs:
                       dict(c='w', ls='-', linewidth=7, alpha=.4), **kwargs)
        return self._plot(plot_result, **kwargs)

    def plot_phase_plane(self, indices=None, **kwargs):
        """ Plots a phase portrait from last integration.

        Parameters
        ----------
        indices : iterable of int
        names : iterable of str
        \*\*kwargs:
            See :func:`pyodesys.plotting.plot_phase_plane`

        """
        return self._plot(plot_phase_plane, indices=indices, **kwargs)

    def plot_invariant_violations(self):
        invar = self.odesys.get_invariants_callback()
        abs_viol = invar(self._internal('xout'), self._internal('yout'), self._internal('params'))
        invar_names = self.odesys.all_invariant_names()
        return self._plot(plot_result, internal_yout=abs_viol, names=invar_names)