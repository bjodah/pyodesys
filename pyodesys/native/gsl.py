# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

from .base import NativeCode, NativeSys, _compile_kwargs

import pygslodeiv2


class NativeGSLCode(NativeCode):
    wrapper_name = '_gsl_wrapper'

    def __init__(self, *args, **kwargs):
        self.compile_kwargs = _compile_kwargs.copy()
        self.compile_kwargs['include_dirs'].append(pygslodeiv2.get_include())
        self.compile_kwargs['libraries'].extend(['gsl', 'gslcblas', 'm'])
        super(NativeGSLCode, self).__init__(*args, **kwargs)


class NativeGSLSys(NativeSys):
    _NativeCode = NativeGSLCode
    _native_name = 'gsl'
