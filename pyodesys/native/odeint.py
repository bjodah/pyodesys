# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import copy

from ..util import import_
from ._base import _NativeCodeBase, _NativeSysBase, _compile_kwargs

pyodeint = import_('pyodeint')


class NativeOdeintCode(_NativeCodeBase):
    wrapper_name = '_odeint_wrapper'

    def __init__(self, *args, **kwargs):
        self.compile_kwargs = copy.deepcopy(_compile_kwargs)
        self.compile_kwargs['include_dirs'].append(pyodeint.get_include())
        self.compile_kwargs['libraries'].extend(['m'])
        super(NativeOdeintCode, self).__init__(*args, **kwargs)


class NativeOdeintSys(_NativeSysBase):
    _NativeCode = NativeOdeintCode
    _native_name = 'odeint'
