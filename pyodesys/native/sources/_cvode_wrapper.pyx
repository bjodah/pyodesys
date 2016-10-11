# -*- mode: cython -*-
# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: extra_compile_args = -std=c++11

from libcpp cimport bool
from libcpp.utility cimport pair
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libcpp.string cimport string

cimport numpy as cnp
from odesys_anyode cimport OdeSys
from odesys_util cimport adaptive_return
from cvodes_cxx cimport lmm_from_name
from cvodes_anyode_nogil cimport simple_predefined, simple_adaptive

import numpy as np


cdef dict _as_dict(unordered_map[string, int] nfo):
    return {str(k.decode('utf-8')): v for k, v in dict(nfo).items()}


cdef pair[pair[vector[double], vector[double]], unordered_map[string, int]] _integrate_adaptive(
    double * y0, double x0, double xend, double * params, vector[double] atol, double rtol,
    string lmm, double dx0, double dx_min, double dx_max, long int mxsteps, bool with_jacobian,
    int iter_type, int linear_solver) nogil except+:
    cdef:
        OdeSys * odesys
        vector[int] root_indices
        pair[vector[double], vector[double]] result
    odesys = new OdeSys(params)
    try:
        result = simple_adaptive[OdeSys](
            odesys, atol, rtol, lmm_from_name(lmm), y0, x0, xend, root_indices,
            mxsteps, dx0, dx_min, dx_max, with_jacobian, iter_type, linear_solver)
        return adaptive_return(result, odesys.last_integration_info)
    finally:
        del odesys


cdef unordered_map[string, int] _integrate_predefined(
    double * y0, size_t nout, double * xout, double * yout, double * params, vector[double] atol,
    double rtol, string lmm, double dx0, double dx_min, double dx_max, long int mxsteps,
    bool with_jacobian, int iter_type, int linear_solver) nogil except+:
    cdef:
        OdeSys * odesys
        vector[int] root_indices
        vector[double] root_out
    odesys = new OdeSys(params)
    try:
        simple_predefined[OdeSys](odesys, atol, rtol, lmm_from_name(lmm), y0, nout, xout, yout, root_indices,
                                  root_out, mxsteps, dx0, dx_min, dx_max, with_jacobian,
                                  iter_type, linear_solver)
        return odesys.last_integration_info
    finally:
        del odesys


def integrate_predefined(cnp.ndarray[cnp.float64_t, ndim=1] y0,
                         cnp.ndarray[cnp.float64_t, ndim=1] xout,
                         cnp.ndarray[cnp.float64_t, ndim=1] params,
                         vector[double] atol,
                         double rtol,
                         double dx0=0.0, double dx_min=0.0, double dx_max=0.0,
                         long int mxsteps=0,
                         int iter_type=0, int linear_solver=0, str method='BDF',
                         bool with_jacobian=True):
    cdef:
        unordered_map[string, int] nfo
        cnp.ndarray[cnp.float64_t, ndim=2] yout
    if np.isnan(y0).any():
        raise ValueError("NaN found in y0")
    yout = np.empty((xout.shape[xout.ndim - 1], y0.shape[y0.ndim - 1]))
    nfo = _integrate_predefined(&y0[0], xout.shape[xout.ndim - 1], &xout[0], &yout[0, 0],
                                <double *>(NULL) if params.size == 0 else &params[0],
                                atol, rtol, method.lower().encode('UTF-8'), dx0, dx_min, dx_max, mxsteps,
                                with_jacobian, iter_type, linear_solver)
    return yout, _as_dict(nfo)


def integrate_adaptive(cnp.ndarray[cnp.float64_t, ndim=1] y0,
                       double x0, double xend,
                       cnp.ndarray[cnp.float64_t, ndim=1] params,
                       vector[double] atol,
                       double rtol,
                       double dx0=0.0, double dx_min=0.0, double dx_max=0.0,
                       long int mxsteps=0,
                       int iter_type=0, int linear_solver=0, str method='BDF',
                       bool with_jacobian=True):
    cdef:
        cdef unordered_map[string, int] nfo
        cdef vector[double] xout, yout
        cnp.ndarray[cnp.float64_t] yout_arr
    if np.isnan(y0).any():
        raise ValueError("NaN found in y0")
    (xout, yout), nfo = _integrate_adaptive(
        &y0[0], x0, xend, <double *>(NULL) if params.size == 0 else &params[0],
        atol, rtol, method.lower().encode('UTF-8'), dx0, dx_min, dx_max, mxsteps, with_jacobian, iter_type, linear_solver)
    yout_arr = np.asarray(yout)
    return np.asarray(xout), yout_arr.reshape((len(xout), y0.shape[y0.ndim - 1])), _as_dict(nfo)
