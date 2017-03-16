# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import sys
import numpy as np


def render_mako(tmpl, **kwargs):
    from mako.template import Template
    from mako.exceptions import text_error_template
    try:
        return str(Template(tmpl).render(**kwargs))
    except:
        sys.stderr.write(text_error_template().render())
        raise


def parse_standalone_output(lines):
    outs = []
    tout, yout, params = None, None, None
    for line in lines:
        if line.startswith('{'):
            outs.append((tout, yout, params, eval(line)))
            tout, yout = None, None
        else:
            if tout is None:
                tout, yout = [], []
                params = line.split()
            else:
                items = line.split()
                tout.append(items[0])
                yout.append(items[1:])
    return [(np.array(_t), np.array(_y), np.asarray(_p), _nfo) for _t, _y, _p, _nfo in outs]
