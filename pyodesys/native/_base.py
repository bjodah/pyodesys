# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

from datetime import datetime as dt
from functools import reduce
import logging
from operator import add
import copy
import os
import shutil
import sys
import tempfile

import sympy
from sympy.codegen.ast import CodeBlock, Assignment
import numpy as np
import pkg_resources

from ..symbolic import SymbolicSys
from .. import __version__

from .symcse.groupwise import GroupwiseCSE

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
    'options': ['warn', 'pic', 'debug', 'openmp'],  # DO-NOT-MERGE!!! debug/fast
    'std': 'c++11',
    'include_dirs': [np.get_include(), pkg_resources.resource_filename(__name__, 'sources')],
    'libraries': [],
    'cplus': True,
}


def get_compile_kwargs():
    kw = copy.deepcopy(_compile_kwargs)
    if options := os.environ.get("PYODESYS_OPTIONS"):
        kw['options'] = options.split(',')
    return kw


_ext_suffix = '.so'  # sysconfig.get_config_var('EXT_SUFFIX')
_obj_suffix = '.o'  # os.path.splitext(_ext_suffix)[0] + '.o'  # '.obj'


class _AssignerBase:
    def all(self, **kwargs):
        return "\n".join(self(i, **kwargs) for i in range(self.n))


def _r(s):
    if isinstance(s, sympy.Symbol):
        return s
    else:
        return sympy.Symbol(s, real=True)


class _AssignerGW(_AssignerBase):
    def __init__(self, k, gw):
        self.k = k
        self.gw = gw
        self.n = len(gw.exprs(k))

    def expr_is_zero(self, i):
        return self.gw.exprs(self.k)[i] == 0

    def __call__(self, i, assign_to=lambda i: _r("out[%s]" % i)):
        return self.gw.render(Assignment(_r(assign_to(i)), self.gw.exprs(self.k)[i]))


class _AssignerPlain(_AssignerBase):
    def __init__(self, k, all_exprs):
        self.k = k
        self.all_exrs = all_exprs
        self.n = len(all_exprs[k])

    def expr_is_zero(self, i):
        return self.all_exprs[i] == 0

    def __call__(self, i, assign_to=lambda i: _r("out[%s]" % i)):
        return self.odesys.be.ccode(
            Assignment(_r(assign_to(i)), self.all_exprs[self.k][i])
        ) + ';'


class _NativeCodeBase(Cpp_Code):
    """Base class for generated code.

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
        'p_realtype': 'double',
        'p_indextype': 'int',
        'p_baseclass': 'OdeSysBase'
    }

    _support_roots = False
    # `namespace_override` is set in init
    # `namespace_extend` is set in init

    def __init__(self, odesys, *args, use_cse=None, **kwargs):
        if Cpp_Code is object:
            raise ModuleNotFoundError("failed to import Cpp_Code from pycodeexport")
        if compile_sources is None:
            raise ModuleNotFoundError("failed to import compile_sources from pycompilation")
        if odesys.nroots > 0 and not self._support_roots:
            raise ValueError("%s does not support nroots > 0" % self.__class__.__name__)
        if use_cse is None:
            use_cse = os.getenv('PYODESYS_NATIVE_CSE', '1') == '1'
        self.use_cse = use_cse
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
        self.compensated_summation = kwargs.pop("compensated_summation", os.environ.get("PYODESYS_COMPENSATED_SUMMATION", "0") == "1")
        super().__init__(*args, logger=logger, **kwargs)

    # def _ccode(self, expr, subsd):
    #     expr_x = expr.xreplace(subsd)
    #     return self.odesys.be.ccode(expr_x)

    def variables(self):
        ny = self.odesys.ny
        if self.odesys.band is not None:
            raise NotImplementedError("Banded jacobian not yet implemented.")

        all_invar = tuple(self.odesys.all_invariants())
        ninvar = len(all_invar)
        jac = self.odesys.get_jac()
        nnz = self.odesys.nnz
        all_exprs = dict(
            rhs=self.odesys.exprs,
            invar=all_invar
        )
        if jac is not False and nnz < 0:
            jac_dfdx = list(reduce(add, jac.tolist() + self.odesys.get_dfdx().tolist()))
            all_exprs["jac_dfdt"] = jac_dfdx
            nj = len(jac_dfdx)
        elif jac is not False and nnz >= 0:
            jac_dfdx = list(reduce(add, jac.tolist()))
            all_exprs["jac_dfdt"] = jac_dfdx
            nj = len(jac_dfdx)
        else:
            nj = 0

        jtimes = self.odesys.get_jtimes()
        if jtimes is not False:
            v, jtimes_exprs = jtimes
            all_exprs["jtimes"] = jtimes_exprs
        else:
            v = ()
            jtimes_exprs = ()

        first_step = self.odesys.first_step_expr
        if first_step is not None:
            all_exprs["first_step"] = [first_step]

        if self.odesys.roots is not None:
            all_exprs["roots"] = self.odesys.roots

        subsd = {k: self.odesys.be.Symbol('y[%d]' % idx) for
                 idx, k in enumerate(self.odesys.dep)}
        if self.odesys.indep is not None:
            subsd[self.odesys.indep] = self.odesys.be.Symbol('x')

        subsd.update({k: self.odesys.be.Symbol('m_p[%d]' % idx) for
                      idx, k in enumerate(self.odesys.params)})

        if jtimes is not False:
            subsd.update({k: self.odesys.be.Symbol('v[%d]' % idx) for
                         idx, k in enumerate(v)})

        if self.use_cse:
            if self.compensated_summation:
                from .symcse.compensated import _NeumaierTransformer as Transformer
            else:
                from .symcse.core import NullTransformer as Transformer
            ignore = (() if self.odesys.indep is None else (self.odesys.indep,)) + self.odesys.dep + v
            gw = GroupwiseCSE(
                all_exprs,
                common_cse_template="m_cse[{}]",
                common_ignore=ignore,
                subsd=subsd,
                Transformer=Transformer,
                pre_process=None
            )

            def not_arr(s):
                return '[' not in s.name

            def _cses(k, assign_to=lambda i: _r("out[%d]" % i)):
                return CodeBlock(*gw.statements(k, declare=not_arr))
            cses = {k: gw.render(_cses(k)) for k in gw.keys}
            n_common_cses = gw.n_remapped
            common_cses = gw.render(CodeBlock(*gw.common_statements(declare=not_arr)))

            assigners = {k: _AssignerGW(k, gw) for k in gw.keys}
        else:
            logger.info("Not using common subexpression elimination (disabled by PYODESYS_NATIVE_CSE)")
            n_common_cses=0
            common_cses=""

            assigners = {k: _AssignerPlain(k, all_exprs) for k in all_exprsself}

        ns = dict(
            _message_for_rendered=[
                "-*- mode: read-only -*-",
                "This file was generated using pyodesys-%s at %s" % (
                    __version__, dt.now().isoformat())
            ],
            p_common={
                'cses': common_cses,
                'n_cses': n_common_cses
            },
            p_odesys=self.odesys,
            p_rhs={
                'cses': cses["rhs"],
                'assign': assigners["rhs"]
            },
            p_jtimes=None if jtimes is False else{
                'cses': cses["jtimes"],
                'assign': assigners["jtimes"]
            },
            p_jac_dense=None if jac is False or nnz >= 0 else {
                'cses': cses["jac_dfdt"],
                'assign': assigners["jac_dfdt"]
            },
            p_jac_sparse=None if jac is False or nnz < 0 else {
                'cses': cses["jac_dfdt"],
                'assign': assigners["jac_dfdt"],
                'colptrs': self.odesys._colptrs,
                'rowvals': self.odesys._rowvals
            },
            p_first_step=None if first_step is None else {
                'cses': cses["first_step"],
                'assign': assigners["first_step"]
            },
            p_roots=None if self.odesys.roots is None else {
                'cses': cses["roots"],
                'assign': assigners["roots"]
            },
            p_invariants=None if all_invar == () else {
                'cses': cses["invar"],
                'assign': assigners["invar"],
                'n_invar': len(all_invar)
            },
            p_nroots=self.odesys.nroots,
            p_constructor=[],
            p_get_dx_max=False
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

    def __init__(self, *args, native_code_kw=None, **kwargs):
        namespace_override = kwargs.pop('namespace_override', {})
        namespace_extend = kwargs.pop('namespace_extend', {})
        save_temp = kwargs.pop('save_temp', False)
        if 'init_indep' not in kwargs:  # we need to trigger append_iv for when invariants are used
            kwargs['init_indep'] = True
            kwargs['init_dep'] = True
        super(_NativeSysBase, self).__init__(*args, **kwargs)
        self._native = self._NativeCode(
            self, save_temp=save_temp,
            namespace_override=namespace_override,
            namespace_extend=namespace_extend,
            **(native_code_kw or {}))

    def integrate(self, *args, **kwargs):
        integrator = kwargs.pop('integrator', 'native')
        if integrator not in ('native', self._native_name):
            raise ValueError("Got incompatible kwargs integrator=%s" % integrator)
        else:
            kwargs['integrator'] = 'native'

        return super().integrate(*args, **kwargs)

    def rhs(self, intern_t, intern_y, intern_p):
        return self._native.mod.rhs(intern_t, intern_y, intern_p)

    def jac(self, intern_t, intern_y, intern_p):
        return self._native.mod.dense_jac_cmaj(intern_t, intern_y, intern_p)

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
            if 'njvev' not in info[idx] and 'krylov_n_jac_times_evals' in info[idx]:
                info[idx]['njvev'] = info[idx]['krylov_n_jac_times_evals']

        return info
