# -*- mode: cython -*-
# -*- coding: utf-8 -*-

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

from anyode cimport Info

cdef extern from "odesys_anyode.hpp" namespace "odesys_anyode":
    cdef cppclass OdeSys[Real_t, Index_t]:
        OdeSys(const Real_t * const, vector[Real_t], Real_t, Real_t,
               bool, Real_t, vector[Real_t]) except + nogil
        Index_t get_ny() nogil
        Real_t get_dx0(Real_t, const Real_t * const) nogil
        unsigned nfev, njev, njvev
        Info current_info
