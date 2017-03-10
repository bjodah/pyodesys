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

    def copy(self):
        return Result(self.xout.copy(), self.yout.copy(), self.params.copy(),
                      self.info.copy(), self.odesys)

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

    def get_param(self, param_name):
        return self.params[self.odesys.param_names.index(param_name)]

    def get_dep(self, name):
        return self.yout[..., self.odesys.names.index(name)]

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
                res, err = res_cub, np.abs(res_cub - np.asarray(res_lin))

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

    def _plot(self, cb, x=None, y=None, legend=None, **kwargs):
        if x is None:
            x = self.xout
        if y is None:
            y = self.yout

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
        if legend is None:
            if kwargs.get('latex_names', None) is not None or kwargs['names'] is not None:
                legend = True
        return cb(x, y, legend=legend, **kwargs)

    def plot(self, info_vlines_kw=None, between=None, deriv=False, title_info=0, **kwargs):
        """ Plots the integrated dependent variables from last integration.

        Parameters
        ----------
        info_vlines_kw : dict
            Keyword arguments passed to :func:`.plotting.info_vlines`,
            an empty dict will be used if `True`. Need to pass `ax` when given.
        indices : iterable of int
        between : length 2 tuple
        deriv : bool
            Plot derivatives (internal variables).
        names : iterable of str
        \*\*kwargs:
            See :func:`pyodesys.plotting.plot_result`
        """
        if between is not None:
            if 'x' in kwargs or 'y' in kwargs:
                raise ValueError("x/y & between given.")
            kwargs['x'], kwargs['y'] = self.between(*between)
        if info_vlines_kw is not None:
            if info_vlines_kw is True:
                info_vlines_kw = {}
            info_vlines(kwargs['ax'], self.xout, self.info, **info_vlines_kw)
            self._plot(plot_result, plot_kwargs_cb=lambda *args, **kwargs:
                       dict(c='w', ls='-', linewidth=7, alpha=.4), **kwargs)
        if deriv:
            if 'y' in kwargs:
                raise ValueError("Cannot give both deriv=True and y.")
            kwargs['y'] = self.odesys.f_cb(self._internal('xout'), self._internal('yout'), self._internal('params'))
        ax = self._plot(plot_result, **kwargs)
        if title_info:
            ax.set_title(
                (self.odesys.description or '') +
                ', '.join(
                    (['%d steps' % self.info['n_steps']] if self.info.get('n_steps', -1) >= 0 else []) +
                    [
                        '%d fev' % self.info['nfev'],
                        '%d jev' % self.info['njev'],
                    ] + ([
                        '%.2g s CPU' % self.info['time_cpu']
                    ] if title_info > 1 and self.info.get('time_cpu', -1) >= 0 else [])
                ) +
                (', success' if self.info['success'] else ', failed'),
                {'fontsize': 'medium'} if title_info > 1 else {}
            )

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

    def plot_invariant_violations(self, **kwargs):
        invar = self.odesys.get_invariants_callback()
        viol = invar(self._internal('xout'), self._internal('yout'), self._internal('params'))
        abs_viol = np.abs(viol - viol[0, :])
        invar_names = self.odesys.all_invariant_names()
        return self._plot(plot_result, x=self._internal('xout'), y=abs_viol, names=invar_names,
                          latex_names=kwargs.pop('latex_names', invar_names), indices=None, **kwargs)

    def extend_by_integration(self, xend, params=None, odesys=None, autonomous=None, **kwargs):
        odesys = odesys or self.odesys
        if autonomous is None:
            autonomous = odesys.autonomous_interface
        x0 = self.xout[-1]
        nx0 = self.xout.size
        res = odesys.integrate((xend - x0) if autonomous else (x0, xend), self.yout[..., -1, :],
                               params or self.params, **kwargs)
        self.xout = np.concatenate((self.xout, res.xout[1:] + (x0 if autonomous else 0)))
        self.yout = np.concatenate((self.yout, res.yout[..., 1:, :]))
        new_info = {k: v for k, v in self.info.items() if not k.startswith('internal')}
        for k, v in res.info.items():
            if k.startswith('internal'):
                continue
            elif k == 'success':
                new_info[k] = new_info[k] and v
            elif k.endswith('_xvals'):
                new_info[k] = np.concatenate((new_info[k], v + (x0 if autonomous else 0)))
            elif k.endswith('_indices'):
                new_info[k].extend([itm + nx0 - 1 for itm in v])
            elif isinstance(v, str):
                if isinstance(new_info[k], str):
                    new_info[k] = [new_info[k]]
                new_info[k].append(v)
            else:
                new_info[k] += v
        self.info = new_info
        return self
