# -*- mode: cython -*-
# -*- coding: utf-8 -*-

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

from anyode cimport Info

cdef extern from "odesys_anyode_iterative.hpp" namespace "odesys_anyode":
    cdef cppclass OdeSys[Real_t, Index_t]:
        OdeSys(const Real_t * const, vector[Real_t], Real_t, Real_t,
               bool, Real_t, vector[Real_t]) nogil except +
        Info current_info
        bool autonomous_exprs
        bool record_rhs_xvals
        bool record_jac_xvals
        bool record_order
        bool record_fpe
