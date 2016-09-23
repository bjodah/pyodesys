# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import os
from .base import NativeCode, NativeSys, _compile_kwargs

import pycvodes


class NativeCvodeCode(NativeCode):
    wrapper_name = '_cvode_wrapper'

    def __init__(self, *args, **kwargs):
        self.compile_kwargs = _compile_kwargs.copy()
        self.compile_kwargs['include_dirs'].append(pycvodes.get_include())
        self.compile_kwargs['libraries'].extend(['sundials_cvodes', os.environ.get('LLAPACK', 'lapack'),
                                                 'sundials_nvecserial', 'm'])
        super(NativeCvodeCode, self).__init__(*args, **kwargs)


class NativeCvodeSys(NativeSys):
    _NativeCode = NativeCvodeCode
    _native_name = 'cvode'
