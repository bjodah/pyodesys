# -*- mode: cython -*-
# -*- coding: utf-8 -*-

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

cdef extern from "odesys_anyode_iterative.hpp" namespace "odesys_anyode":
    cdef cppclass OdeSys:
        OdeSys(const double * const, vector[double], double, double,
               bool, double, vector[double]) nogil except +
        Info current_info        
        bool autonomous_exprs
        bool record_rhs_xvals
        bool record_jac_xvals
        bool record_order
        bool record_fpe
