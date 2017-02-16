# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import copy
import os

try:
    import pycvodes
except ImportError:
    pycvodes = None
else:
    from pycvodes import _config

from ._base import _NativeCodeBase, _NativeSysBase, _compile_kwargs
from .util import render_mako

class NativeCvodeCode(_NativeCodeBase):
    wrapper_name = '_cvode_wrapper'

    namespace = {
        'p_includes': ['"odesys_anyode_iterative.hpp"'],
        'p_support_recoverable_error': True,
        'p_jacobian_set_to_zero_by_solver': True
    }
    _support_roots = True

    def __init__(self, *args, **kwargs):
        self.compile_kwargs = copy.deepcopy(_compile_kwargs)
        self.compile_kwargs['include_dirs'].append(pycvodes.get_include())
        self.compile_kwargs['libraries'].extend(_config.env['SUNDIALS_LIBS'].split(','))
        self.compile_kwargs['libraries'].extend(os.environ.get(
            'PYODESYS_LAPACK', _config.env['LAPACK']).split(','))
        super(NativeCvodeCode, self).__init__(*args, **kwargs)


class NativeCvodeSys(_NativeSysBase):
    _NativeCode = NativeCvodeCode
    _native_name = 'cvode'

    def as_standalone(self, outdir='.'):
        from pycompilation.compilation import src2obj, link
        from pycodeexport.util import render_mako_template_to
        f = render_mako_template_to(os.path.join(os.path.dirname(__file__), 'sources/standalone_template.cpp'),
                                    'standalone.cpp', {'p_odesys': self})
        objf = src2obj(f, **self._native.compile_kwargs)
        return link([os.path.join(self._native._tempdir, self._native.obj_files[0]), objf], **self._native.compile_kwargs)
