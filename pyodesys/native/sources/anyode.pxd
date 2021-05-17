# -*- coding: utf-8; mode: cython -*-

from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "anyode/anyode.hpp" namespace "AnyODE":
    cdef cppclass OdeSysBase[Real_t, Index_t]:
        int nfev, njev, njvev
        bool use_get_dx_max

    cdef cppclass Info:
        unordered_map[string, int] nfo_int
        unordered_map[string, double] nfo_dbl
        unordered_map[string, vector[double]] nfo_vecdbl
        unordered_map[string, vector[int]] nfo_vecint

    cdef cppclass Status:
        pass  # Status is an enum class

cdef extern from "anyode/anyode.hpp" namespace "AnyODE::Status":
    cdef Status success
    cdef Status recoverable_error
    cdef Status unrecoverable_error
