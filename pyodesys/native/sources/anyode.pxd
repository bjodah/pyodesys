# -*- coding: utf-8; mode: cython -*-

from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "anyode/anyode.hpp" namespace "AnyODE":
     cdef cppclass OdeSysBase[T]:
         int nfev, njev
         bool use_get_dx_max

cdef extern from "anyode/anyode.hpp" namespace "AnyODE":
     cdef cppclass Info:
        unordered_map[string, int] nfo_int
        unordered_map[string, double] nfo_dbl
        unordered_map[string, vector[double]] nfo_vecdbl
        unordered_map[string, vector[int]] nfo_vecint
