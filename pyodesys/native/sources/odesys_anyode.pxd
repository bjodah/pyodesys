# -*- mode: cython -*-
# -*- coding: utf-8 -*-

from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string

cdef extern from "odesys_anyode.hpp":
    cdef cppclass OdeSys:
        OdeSys(const double * const) nogil except +
        int get_ny()
        unordered_map[string, int] last_integration_info
        size_t m_nfev, m_njev

    cdef pair[pair[vector[double], vector[double]], unordered_map[string, int]] adaptive_return(
        pair[vector[double], vector[double]], unordered_map[string, int]
    ) nogil
