# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

from .base import NativeCode, NativeSys, _compile_kwargs

import pyodeint


class NativeOdeintCode(NativeCode):
    wrapper_name = '_odeint_wrapper'

    def __init__(self, *args, **kwargs):
        self.compile_kwargs = _compile_kwargs.copy()
        self.compile_kwargs['include_dirs'].append(pyodeint.get_include())
        self.compile_kwargs['libraries'].extend(['m'])
        super(NativeOdeintCode, self).__init__(*args, **kwargs)


class NativeOdeintSys(NativeSys):
    _NativeCode = NativeOdeintCode
    _native_name = 'odeint'
