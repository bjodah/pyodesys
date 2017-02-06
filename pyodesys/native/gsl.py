# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import copy
import os

try:
    import pygslodeiv2
except ImportError:
    pygslodeiv2 = None
else:
    from pygslodeiv2 import _config

from ._base import _NativeCodeBase, _NativeSysBase, _compile_kwargs


class NativeGSLCode(_NativeCodeBase):
    """ Looks for the environment variable: ``PYODESYS_BLAS`` (``gslcblas``) """
    wrapper_name = '_gsl_wrapper'

    def __init__(self, *args, **kwargs):
        self.compile_kwargs = copy.deepcopy(_compile_kwargs)
        self.compile_kwargs['include_dirs'].append(pygslodeiv2.get_include())
        self.compile_kwargs['libraries'].extend(_config.env['GSL_LIBS'].split(','))
        self.compile_kwargs['libraries'].extend(os.environ.get('PYODESYS_BLAS', _config.env['BLAS']).split(','))
        super(NativeGSLCode, self).__init__(*args, **kwargs)


class NativeGSLSys(_NativeSysBase):
    _NativeCode = NativeGSLCode
    _native_name = 'gsl'
