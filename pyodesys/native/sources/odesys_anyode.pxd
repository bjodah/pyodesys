# -*- mode: cython -*-
# -*- coding: utf-8 -*-

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

from anyode cimport Info

cdef extern from "odesys_anyode.hpp" namespace "odesys_anyode":
    cdef cppclass OdeSys:
        OdeSys(const double * const, vector[double], double, double,
               bool, double, vector[double]) nogil except +
        int get_ny() nogil
        double get_dx0(double, const double * const) nogil
        unsigned nfev, njev
        Info current_info        
