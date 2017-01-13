# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)


class Result(object):

    def __init__(self, xout, yout, info, odesys):
        self.xout = xout
        self.yout = yout
        self.info = info
        self.odesys = odesys

    def __len__(self):
        return 3

    def __getitem__(self, key):
        if key == 0:
            return self.xout
        elif key == 1:
            return self.yout
        elif key == 2:
            return self.info
        elif key == 3:
            raise StopIteration
        else:
            raise KeyError("Invalid key: %s (for backward compatibility reasons)." % str(key))
