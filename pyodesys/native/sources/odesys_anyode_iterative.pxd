# -*- mode: cython -*-
# -*- coding: utf-8 -*-

from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

cdef extern from "odesys_anyode_iterative.hpp" namespace "odesys_anyode":
    cdef cppclass OdeSys:
        OdeSys(const double * const, vector[double], double) nogil except +
        unordered_map[string, int] last_integration_info
        unordered_map[string, double] last_integration_info_dbl