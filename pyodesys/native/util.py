# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import sys


def render_mako(tmpl, **kwargs):
    from mako.template import Template
    from mako.exceptions import text_error_template
    try:
        return str(Template(tmpl).render(**kwargs))
    except:
        sys.stderr.write(text_error_template().render())
        raise
