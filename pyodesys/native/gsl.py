# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import copy
import os
import warnings

try:
    import pygslodeiv2
except ImportError:
    pygslodeiv2 = None

from .base import _NativeCodeBase, _NativeSysBase, _compile_kwargs


class NativeGSLCode(_NativeCodeBase):
    """ Looks for the environment variable: ``LBLAS`` (``gslcblas``) """
    wrapper_name = '_gsl_wrapper'

    def __init__(self, *args, **kwargs):
        self.compile_kwargs = copy.deepcopy(_compile_kwargs)
        self.compile_kwargs['include_dirs'].append(pygslodeiv2.get_include())
        self.compile_kwargs['libraries'].extend(['gsl'])

        LBLAS = os.environ.get('LBLAS')
        # LLAPACK = os.environ.get('LLAPACK', 'lapack')
        if LBLAS is None:
            self.compile_kwargs['libraries'].append('gslcblas')
        elif LBLAS == '':
            # if LLAPACK in ('lapack', ''):
            warnings.warn("Are you sure you are linking with BLAS?")
        else:
            self.compile_kwargs['libraries'].append(LBLAS)
        # self.compile_kwargs['libraries'].extend([LLAPACK, 'm'])
        super(NativeGSLCode, self).__init__(*args, **kwargs)


class NativeGSLSys(_NativeSysBase):
    _NativeCode = NativeGSLCode
    _native_name = 'gsl'
