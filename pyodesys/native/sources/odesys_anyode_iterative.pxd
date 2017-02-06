# -*- mode: cython -*-
# -*- coding: utf-8 -*-

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

cdef extern from "odesys_anyode_iterative.hpp" namespace "odesys_anyode":
    cdef cppclass OdeSys:
        OdeSys(const double * const, vector[double], double, double, bool) nogil except +
        unordered_map[string, int] last_integration_info
        unordered_map[string, double] last_integration_info_dbl
        unordered_map[string, vector[double]] last_integration_info_vecdbl
        unordered_map[string, vector[int]] last_integration_info_vecint
        bool record_rhs_xvals
        bool record_jac_xvals
        bool record_order
        bool record_fpe
