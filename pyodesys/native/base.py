# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

from functools import reduce
import logging
from operator import add
import os
import shutil
import sys
import tempfile

from appdirs import user_cache_dir
import numpy as np
import pkg_resources
from pycodeexport.codeexport import Cpp_Code
from pycompilation import compile_sources

from ..symbolic import SymbolicSys
from .. import __version__

appauthor = "bjodah"
appname = "python%d.%d-pyodesys-%s" % (sys.version_info[:2] + (__version__,))
cachedir = user_cache_dir(appname, appauthor)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

_compile_kwargs = {
    'options': ['warn', 'pic', 'fast', 'openmp'],
    'std': 'c++11',
    'include_dirs': [np.get_include(),
                     pkg_resources.resource_filename(__name__, 'sources')],
    'libraries': [],
    'cplus': True,
}

_ext_suffix = '.so'  # sysconfig.get_config_var('EXT_SUFFIX')
_obj_suffix = '.o'  # os.path.splitext(_ext_suffix)[0] + '.o'  # '.obj'


class NativeCode(Cpp_Code):
    wrapper_name = None
    basedir = os.path.dirname(__file__)
    templates = ('sources/odesys_anyode_template.cpp',)
    build_files = ()  # ('sources/odesys_anyode.hpp', ...)
    source_files = ('odesys_anyode.cpp',)
    obj_files = ('odesys_anyode.o',)

    def __init__(self, odesys, *args, **kwargs):
        self.obj_files = self.obj_files + ('%s%s' % (self.wrapper_name, _obj_suffix),)
        self.so_file = '%s%s' % (self.wrapper_name, '.so')
        _wrapper_src = pkg_resources.resource_filename(
            __name__, 'sources/%s.pyx' % self.wrapper_name)
        _wrapper_obj = os.path.join(cachedir, '%s%s' % (self.wrapper_name, _obj_suffix))
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)
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
        super(NativeCode, self).__init__(*args, logger=logger, **kwargs)

    def variables(self):
        ny = self.odesys.ny
        if self.odesys.band is not None:
            raise NotImplementedError("Banded jacobian not yet implemented.")

        subsd = {k: self.odesys.be.Symbol('y[%d]' % idx) for
                 idx, k in enumerate(self.odesys.dep)}
        subsd[self.odesys.indep] = self.odesys.be.Symbol('t')
        subsd.update({k: self.odesys.be.Symbol('m_p[%d]' % idx) for
                      idx, k in enumerate(self.odesys.params)})

        def _ccode(expr):
            return self.odesys.be.ccode(expr.xreplace(subsd))

        rhs_cses, rhs_exprs = self.odesys.be.cse(self.odesys.exprs)
        jac_cses, jac_exprs = self.odesys.be.cse(list(reduce(
            add, self.odesys.get_jac().tolist() + self.odesys.get_dfdx().tolist())))

        ns = dict(
            _message_for_rendered='This is a rendered source file (from template).',
            p_odesys=self.odesys,
            p_rhs_cses=[(symb.name, _ccode(expr)) for symb, expr in rhs_cses],
            p_rhs_exprs=map(_ccode, rhs_exprs),
            p_jac_cses=[(symb.name, _ccode(expr)) for symb, expr in jac_cses],
            p_jac_exprs={(idx//ny, idx % ny): _ccode(expr)
                         for idx, expr in enumerate(jac_exprs[:ny*ny])},
            p_jac_dfdt_exprs=list(map(_ccode, jac_exprs[ny*ny:]))
        )
        return ns


class NativeSys(SymbolicSys):

    _NativeCode = None
    _native_name = None

    def __init__(self, *args, **kwargs):
        super(NativeSys, self).__init__(*args, **kwargs)
        self._native = self._NativeCode(self)

    def integrate(self, *args, **kwargs):
        integrator = kwargs.pop('integrator', 'native')
        if integrator not in ('native', self._native_name):
            raise ValueError("Got incompatible kwargs integrator=%s" % integrator)
        else:
            kwargs['integrator'] = 'native'

        return super(NativeSys, self).integrate(*args, **kwargs)

    def _integrate_native(self, intern_xout, intern_y0, intern_p, force_predefined=False,
                          atol=1e-8, rtol=1e-8, nsteps=500, first_step=0.0, **kwargs):
        atol = np.atleast_1d(atol)
        y0 = np.ascontiguousarray(intern_y0, dtype=np.float64)
        params = np.ascontiguousarray(intern_p, dtype=np.float64)
        if atol.size != 1 and atol.size != self.ny:
            raise ValueError("atol needs to be of length 1 or %d" % self.ny)
        if intern_xout.shape[-1] == 2 and not force_predefined:
            intern_xout, yout, info = self._native.mod.integrate_adaptive(
                y0=y0,
                x0=np.ascontiguousarray(intern_xout[:, 0], dtype=np.float64),
                xend=np.ascontiguousarray(intern_xout[:, 1], dtype=np.float64),
                params=params, atol=atol, rtol=rtol,
                mxsteps=nsteps, dx0=first_step, **kwargs)
        else:
            yout, info = self._native.mod.integrate_predefined(
                y0=y0, xout=np.ascontiguousarray(intern_xout, dtype=np.float64),
                params=params, atol=atol, rtol=rtol,
                mxsteps=nsteps, dx0=first_step, **kwargs)
        for idx in range(len(info)):
            info[idx]['internal_xout'] = intern_xout[idx]
            info[idx]['internal_yout'] = yout[idx]
            info[idx]['success'] = True
            if 'nfev' not in info[idx] and 'n_rhs_evals' in info[idx]:
                info[idx]['nfev'] = info[idx]['n_rhs_evals']
            if 'njev' not in info[idx] and 'dense_n_dls_jac_evals' in info[idx]:
                info[idx]['njev'] = info[idx]['dense_n_dls_jac_evals']
        return info
