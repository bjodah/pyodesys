# -*- mode: cython -*-
# -*- coding: utf-8 -*-

from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string

cdef extern from "odesys_util.hpp" namespace "odesys_util":
    cdef pair[pair[vector[double], vector[double]], unordered_map[string, int]] adaptive_return(
        pair[vector[double], vector[double]], unordered_map[string, int]
    ) nogil
