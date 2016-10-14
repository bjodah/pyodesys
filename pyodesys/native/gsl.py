# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import copy

import pygslodeiv2

from .base import _NativeCodeBase, _NativeSysBase, _compile_kwargs


class NativeGSLCode(_NativeCodeBase):
    wrapper_name = '_gsl_wrapper'

    def __init__(self, *args, **kwargs):
        self.compile_kwargs = copy.deepcopy(_compile_kwargs)
        self.compile_kwargs['include_dirs'].append(pygslodeiv2.get_include())
        self.compile_kwargs['libraries'].extend(['gsl', 'gslcblas', 'm'])
        super(NativeGSLCode, self).__init__(*args, **kwargs)


class NativeGSLSys(_NativeSysBase):
    _NativeCode = NativeGSLCode
    _native_name = 'gsl'
