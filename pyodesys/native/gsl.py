# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import copy
import os
from ..util import import_
from ._base import _NativeCodeBase, _NativeSysBase, get_compile_kwargs

_config, get_include = import_('pygslodeiv2', '_config', 'get_include')


class NativeGSLCode(_NativeCodeBase):
    """ Looks for the environment variable: ``PYODESYS_BLAS`` (``gslcblas``) """
    wrapper_name = '_gsl_wrapper'

    def __init__(self, *args, **kwargs):
        self.compile_kwargs = get_compile_kwargs(kwargs)
        self.compile_kwargs['include_dirs'].append(get_include())
        self.compile_kwargs['libraries'].extend(_config.env['GSL_LIBS'].split(','))
        self.compile_kwargs['libraries'].extend(os.environ.get('PYODESYS_BLAS', _config.env['BLAS']).split(','))
        super(NativeGSLCode, self).__init__(*args, **kwargs)


class NativeGSLSys(_NativeSysBase):
    _NativeCode = NativeGSLCode
    _native_name = 'gsl'
