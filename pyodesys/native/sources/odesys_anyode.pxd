# -*- mode: cython -*-
# -*- coding: utf-8 -*-

from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string

cdef extern from "odesys_anyode.hpp" namespace "odesys_anyode":
    cdef cppclass OdeSys:
        OdeSys(const double * const) nogil except +
        int get_ny() nogil
        unordered_map[string, int] last_integration_info
        unsigned nfev, njev
