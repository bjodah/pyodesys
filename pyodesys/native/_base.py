# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

from datetime import datetime as dt
from functools import reduce
import logging
from operator import add
import os
import shutil
import sys
import tempfile


import numpy as np
import pkg_resources

from ..symbolic import SymbolicSys
from .. import __version__

try:
    import appdirs
except ImportError:
    cachedir = None
else:
    appauthor = "bjodah"
    appname = "python%d.%d-pyodesys-%s" % (sys.version_info[:2] + (__version__,))
    cachedir = appdirs.user_cache_dir(appname, appauthor)

try:
    from pycodeexport.codeexport import Cpp_Code
except ImportError:
    Cpp_Code = object
    compile_sources = None
else:
    from pycompilation import compile_sources


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

_compile_kwargs = {
    'options': ['warn', 'pic', 'fast', 'openmp'],
    'std': 'c++14',
    'include_dirs': [np.get_include(), pkg_resources.resource_filename(__name__, 'sources')],
    'libraries': [],
    'cplus': True,
}

_ext_suffix = '.so'  # sysconfig.get_config_var('EXT_SUFFIX')
_obj_suffix = '.o'  # os.path.splitext(_ext_suffix)[0] + '.o'  # '.obj'


class _NativeCodeBase(Cpp_Code):
    """ Base class for generated code.

    Note kwargs ``namespace_override`` which allows the user to customize
    the variables used when rendering the template.
    """

    wrapper_name = None
    basedir = os.path.dirname(__file__)
    templates = ('sources/odesys_anyode_template.cpp',)
    _written_files = ()
    build_files = ()
    source_files = ('odesys_anyode.cpp',)
    obj_files = ('odesys_anyode.o',)
    _save_temp = False

    namespace_default = {'p_anon': None}
    namespace = {
        'p_includes': ['"odesys_anyode.hpp"'],
        'p_support_recoverable_error': False,
        'p_jacobian_set_to_zero_by_solver': False,
    }
    _support_roots = False
    # `namespace_override` is set in init
    # `namespace_extend` is set in init

    def __init__(self, odesys, *args, **kwargs):
        if odesys.nroots > 0 and not self._support_roots:
            raise ValueError("%s does not support nroots > 0" % self.__class__.__name__)
        self.namespace_override = kwargs.pop('namespace_override', {})
        self.namespace_extend = kwargs.pop('namespace_extend', {})
        self.tempdir_basename = '_pycodeexport_pyodesys_%s' % self.__class__.__name__
        self.obj_files = self.obj_files + ('%s%s' % (self.wrapper_name, _obj_suffix),)
        self.so_file = '%s%s' % (self.wrapper_name, '.so')
        _wrapper_src = pkg_resources.resource_filename(
            __name__, 'sources/%s.pyx' % self.wrapper_name)
        if cachedir is None:
            raise ImportError("No module named appdirs (needed for caching). Install 'appdirs' using e.g. pip/conda.")
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)
        _wrapper_src = os.path.join(cachedir, '%s%s' % (self.wrapper_name, '.pyx'))
        shutil.copy(pkg_resources.resource_filename(__name__, 'sources/%s.pyx' % self.wrapper_name),
                    _wrapper_src)
        _wrapper_obj = os.path.join(cachedir, '%s%s' % (self.wrapper_name, _obj_suffix))
        prebuild = {_wrapper_src: _wrapper_obj}

        self.build_files = self.build_files + tuple(prebuild.values())

        self.odesys = odesys
        for _src, _dest in prebuild.items():
            if not os.path.exists(_dest):
                tmpdir = tempfile.mkdtemp()
                try:
                    compile_sources([_src], cwd=tmpdir, metadir=cachedir,
                                    logger=logger, **self.compile_kwargs)
                    shutil.copy(os.path.join(tmpdir, os.path.basename(_src)[:-4] + '.o'),
                                _dest)
                finally:
                    if not kwargs.get('save_temp', False):
                        shutil.rmtree(tmpdir)
                if not os.path.exists(_dest):
                    raise OSError("Failed to place prebuilt file at: %s" % _dest)
        super(_NativeCodeBase, self).__init__(*args, logger=logger, **kwargs)

    def variables(self):
        ny = self.odesys.ny
        if self.odesys.band is not None:
            raise NotImplementedError("Banded jacobian not yet implemented.")

        subsd = {k: self.odesys.be.Symbol('y[%d]' % idx) for
                 idx, k in enumerate(self.odesys.dep)}
        subsd[self.odesys.indep] = self.odesys.be.Symbol('x')
        subsd.update({k: self.odesys.be.Symbol('m_p[%d]' % idx) for
                      idx, k in enumerate(self.odesys.params)})

        def _ccode(expr):
            return self.odesys.be.ccode(expr.xreplace(subsd))

        all_invar = tuple(self.odesys.all_invariants())
        jac = self.odesys.get_jac()
        if jac is False:
            all_exprs = self.odesys.exprs + all_invar
        else:
            jac_dfdx = list(reduce(add, jac.tolist() + self.odesys.get_dfdx().tolist()))
            all_exprs = self.odesys.exprs + tuple(jac_dfdx) + all_invar

        def common_cse_symbols():
            idx = 0
            while True:
                yield self.odesys.be.Symbol('m_p_cse[%d]' % idx)
                idx += 1

        if os.getenv('PYODESYS_NATIVE_CSE', '1') == '1':
            cse_cb = self.odesys.be.cse
        else:
            logger.info("Not using common subexpression elimination (disabled by PYODESYS_NATIVE_CSE)")
            cse_cb = lambda exprs, **kwargs: ([], exprs)

        try:
            common_cses, common_exprs = cse_cb(
                all_exprs, symbols=self.odesys.be('cse'),
                ignore=(self.odesys.indep,) + self.odesys.dep)
        except TypeError:  # old version of SymPy does not support ``ignore``
            common_cses, common_exprs = [], all_exprs
        common_cse_subs = {}
        comm_cse_symbs = common_cse_symbols()
        for symb, subexpr in common_cses:
            for expr in common_exprs:
                if symb in expr.free_symbols:
                    common_cse_subs[symb] = next(comm_cse_symbs)
                    break
        common_cses = [(x.xreplace(common_cse_subs), expr.xreplace(common_cse_subs))
                       for x, expr in common_cses]
        common_exprs = [expr.xreplace(common_cse_subs) for expr in common_exprs]

        rhs_cses, rhs_exprs = cse_cb(
            common_exprs[:len(self.odesys.exprs)],
            symbols=self.odesys.be.numbered_symbols('cse'))

        if jac is not False:
            jac_cses, jac_exprs = cse_cb(
                common_exprs[len(self.odesys.exprs):
                             len(self.odesys.exprs)+len(jac_dfdx)],
                symbols=self.odesys.be.numbered_symbols('cse'))

        first_step = self.odesys.first_step_expr
        if first_step is not None:
            first_step_cses, first_step_exprs = cse_cb(
                [first_step],
                symbols=self.odesys.be.numbered_symbols('cse'))

        if self.odesys.roots is not None:
            roots_cses, roots_exprs = cse_cb(
                self.odesys.roots,
                symbols=self.odesys.be.numbered_symbols('cse'))
        if all_invar:
            invar_cses, invar_exprs = cse_cb(
                common_exprs[len(self.odesys.exprs)+len(jac_dfdx):
                             len(self.odesys.exprs)+len(jac_dfdx)+len(all_invar)],
                symbols=self.odesys.be.numbered_symbols('cse')
            )

        ns = dict(
            _message_for_rendered=[
                "-*- mode: read-only -*-",
                "This file was generated using pyodesys-%s at %s" % (
                    __version__, dt.now().isoformat())
            ],
            p_odesys=self.odesys,
            p_common={
                'cses': [(symb.name, _ccode(expr)) for symb, expr in common_cses],
                'nsubs': len(common_cse_subs)
            },
            p_rhs={
                'cses': [(symb.name, _ccode(expr)) for symb, expr in rhs_cses],
                'exprs': list(map(_ccode, rhs_exprs))
            },
            p_jac=None if jac is False else {
                'cses': [(symb.name, _ccode(expr)) for symb, expr in jac_cses],
                'exprs': {(idx//ny, idx % ny): _ccode(expr)
                          for idx, expr in enumerate(jac_exprs[:ny*ny])},
                'dfdt_exprs': list(map(_ccode, jac_exprs[ny*ny:]))
            },
            p_first_step=None if first_step is None else {
                'cses': first_step_cses,
                'expr': _ccode(first_step_exprs[0]),
            },
            p_roots=None if self.odesys.roots is None else {
                'cses': [(symb.name, _ccode(expr)) for symb, expr in roots_cses],
                'exprs': list(map(_ccode, roots_exprs))
            },
            p_invariants=None if all_invar == () else {
                'cses': [(symb.name, _ccode(expr)) for symb, expr in invar_cses],
                'exprs': list(map(_ccode, invar_exprs))
            },
            p_nroots=self.odesys.nroots,
            p_constructor=[],
            p_destructor=[],
            p_get_dx_max=False,
            p_y_preprocessing=None,
        )
        ns.update(self.namespace_default)
        ns.update(self.namespace)
        ns.update(self.namespace_override)
        for k, v in self.namespace_extend.items():
            ns[k].extend(v)
        return ns


class _NativeSysBase(SymbolicSys):

    _NativeCode = None
    _native_name = None

    def __init__(self, *args, **kwargs):
        namespace_override = kwargs.pop('namespace_override', {})
        namespace_extend = kwargs.pop('namespace_extend', {})
        if 'init_indep' not in kwargs:  # we need to trigger append_iv for when invariants are used
            kwargs['init_indep'] = True
            kwargs['init_dep'] = True
        super(_NativeSysBase, self).__init__(*args, **kwargs)
        self._native = self._NativeCode(self,
                                        namespace_override=namespace_override,
                                        namespace_extend=namespace_extend)

    def integrate(self, *args, **kwargs):
        integrator = kwargs.pop('integrator', 'native')
        if integrator not in ('native', self._native_name):
            raise ValueError("Got incompatible kwargs integrator=%s" % integrator)
        else:
            kwargs['integrator'] = 'native'

        return super(_NativeSysBase, self).integrate(*args, **kwargs)

    def _integrate_native(self, intern_x, intern_y0, intern_p, force_predefined=False,
                          atol=1e-8, rtol=1e-8, nsteps=500, first_step=0.0, **kwargs):
        atol = np.atleast_1d(atol)
        y0 = np.ascontiguousarray(intern_y0, dtype=np.float64)
        params = np.ascontiguousarray(intern_p, dtype=np.float64)
        if atol.size != 1 and atol.size != self.ny:
            raise ValueError("atol needs to be of length 1 or %d" % self.ny)

        if intern_x.shape[-1] == 2 and not force_predefined:
            intern_xout, yout, info = self._native.mod.integrate_adaptive(
                y0=y0,
                x0=np.ascontiguousarray(intern_x[:, 0], dtype=np.float64),
                xend=np.ascontiguousarray(intern_x[:, 1], dtype=np.float64),
                params=params, atol=atol, rtol=rtol,
                mxsteps=nsteps, dx0=first_step, **kwargs)
        else:
            yout, info = self._native.mod.integrate_predefined(
                y0=y0, xout=np.ascontiguousarray(intern_x, dtype=np.float64),
                params=params, atol=atol, rtol=rtol,
                mxsteps=nsteps, dx0=first_step, **kwargs)
            intern_xout = intern_x
        for idx in range(len(info)):
            info[idx]['internal_xout'] = intern_xout[idx]
            info[idx]['internal_yout'] = yout[idx]
            info[idx]['internal_params'] = intern_p[idx, ...]
            if 'nfev' not in info[idx] and 'n_rhs_evals' in info[idx]:
                info[idx]['nfev'] = info[idx]['n_rhs_evals']
            if 'njev' not in info[idx] and 'dense_n_dls_jac_evals' in info[idx]:
                info[idx]['njev'] = info[idx]['dense_n_dls_jac_evals']

        return info
