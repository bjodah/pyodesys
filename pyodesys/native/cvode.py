# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import copy
import os
import sys

from ..util import import_
from ._base import _NativeCodeBase, _NativeSysBase, _compile_kwargs

get_include, config, _libs = import_("pycvodes", "get_include", "config", "_libs")

if sys.version_info < (3, 6, 0):
    class ModuleNotFoundError(ImportError):
        pass


class NativeCvodeCode(_NativeCodeBase):
    wrapper_name = '_cvode_wrapper'

    try:
        _realtype = config['REAL_TYPE']
        _indextype = config['INDEX_TYPE']
    except ModuleNotFoundError:
        _realtype = '#error "realtype_failed-to-import-pycvodes-or-too-old-version"'
        _indextype = '#error "indextype_failed-to-import-pycvodes-or-too-old-version"'

    namespace = {
        'p_includes': ['"odesys_anyode_iterative.hpp"'],
        'p_support_recoverable_error': True,
        'p_jacobian_set_to_zero_by_solver': True,
        'p_baseclass': 'OdeSysIterativeBase',
        'p_realtype': _realtype,
        'p_indextype': _indextype
    }
    _support_roots = True

    def __init__(self, *args, **kwargs):
        self.compile_kwargs = copy.deepcopy(_compile_kwargs)
        self.compile_kwargs['define'] = ['PYCVODES_NO_KLU={}'.format("0" if config.get('KLU', True) else "1"),
                                         'PYCVODES_NO_LAPACK={}'.format("0" if config.get('LAPACK', True) else "1"),
                                         'ANYODE_NO_LAPACK={}'.format("0" if config.get('LAPACK', True) else "1")]
        self.compile_kwargs['include_dirs'].append(get_include())
        self.compile_kwargs['libraries'].extend(_libs.get_libs().split(','))
        self.compile_kwargs['libraries'].extend([l for l in os.environ.get(
            'PYODESYS_LAPACK', "lapack,blas" if config["LAPACK"] else "").split(",") if l != ""])
        super(NativeCvodeCode, self).__init__(*args, **kwargs)


class NativeCvodeSys(_NativeSysBase):
    _NativeCode = NativeCvodeCode
    _native_name = 'cvode'

    def as_standalone(self, out_file=None, compile_kwargs=None):
        from pycompilation.compilation import src2obj, link
        from pycodeexport.util import render_mako_template_to
        compile_kwargs = compile_kwargs or {}
        impl_src = open([f for f in self._native._written_files if f.endswith('.cpp')][0], 'rt').read()
        f = render_mako_template_to(
            os.path.join(os.path.dirname(__file__), 'sources/standalone_template.cpp'),
            '%s.cpp' % out_file, {'p_odesys': self, 'p_odesys_impl': impl_src})
        kw = copy.deepcopy(self._native.compile_kwargs)
        kw.update(compile_kwargs)
        objf = src2obj(f, **kw)
        kw['libraries'].append('boost_program_options')
        return link([objf], out_file, **kw)
