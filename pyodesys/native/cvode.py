# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import copy
import os

import pycvodes
from pycvodes import _config

from .base import _NativeCodeBase, _NativeSysBase, _compile_kwargs


class NativeCvodeCode(_NativeCodeBase):
    wrapper_name = '_cvode_wrapper'

    namespace = {
        'p_includes': ['"odesys_anyode_iterative.hpp"'],
        'support_recoverable_error': True
    }

    def __init__(self, *args, **kwargs):
        self.compile_kwargs = copy.deepcopy(_compile_kwargs)
        self.compile_kwargs['include_dirs'].append(pycvodes.get_include())
        _lapack = os.environ.get('PYODESYS_LAPACK', _config.env['LAPACK'])
        self.compile_kwargs['libraries'].extend(['sundials_cvodes', _lapack, 'sundials_nvecserial', 'm'])
        super(NativeCvodeCode, self).__init__(*args, **kwargs)


class NativeCvodeSys(_NativeSysBase):
    _NativeCode = NativeCvodeCode
    _native_name = 'cvode'
