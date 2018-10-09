# -*- mode: cython -*-
# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: extra_compile_args = -std=c++11 -fopenmp
# distutils: extra_link_args = -fopenmp

from collections import Iterable

from libc.stdlib cimport malloc, free
from libcpp cimport bool
from libcpp.utility cimport pair
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libcpp.string cimport string

from odesys_anyode_iterative cimport OdeSys
from cvodes_cxx cimport lmm_from_name, iter_type_from_name, linear_solver_from_name
from cvodes_anyode_parallel cimport multi_predefined, multi_adaptive

import numpy as np
cimport numpy as cnp

cnp.import_array()  # Numpy C-API initialization

# Need these here rather than as imports so that
# typedefs are available at compile- (not run-) time.
# NOTE: base types of "int" and "float" are just
# appropriately-close standins as per Cython rules; will
# be replaced with the exact extern typedef at compile-time.
cdef extern from "cvodes_cxx.hpp":
     ctypedef double realtype
     ctypedef int indextype

# These need to be available as type objects at run type, in addition to the corresponding
# type tags (e.g. np.float64_t), which only exist at compile time and cannot be used with
# np.asarray(..., dtype=)
if sizeof(realtype) == sizeof(cnp.npy_double):
    dtype = np.float64
elif sizeof(realtype) == sizeof(cnp.npy_float):
    dtype = np.float32
elif sizeof(realtype) == sizeof(cnp.npy_longdouble):
    dtype = np.longdouble
else:
    dtype = np.float64

# signature in python methods should be able to accept any floating type regardless
# of what realtype is under the hood. scalars of type "floating" passed to the cython wrapper
# should be auto-cast to realtype when passed to C functions; any vectors/arrays
# will be manually cast below
ctypedef fused floating:
    cnp.float32_t
    cnp.float64_t
    cnp.longdouble_t

ctypedef OdeSys[realtype, indextype] CvodesOdeSys

from odesys_util cimport adaptive_return


def _as_dict(unordered_map[string, int] nfo,
             unordered_map[string, double] nfo_dbl,
             unordered_map[string, vector[double]] nfo_vecdbl,
             unordered_map[string, vector[int]] nfo_vecint,
             root_indices, bool success, root_out=None, mode=None, nreached=None):
    dct = {str(k.decode('utf-8')): v for k, v in dict(nfo).items()}
    dct.update({str(k.decode('utf-8')): v for k, v in dict(nfo_dbl).items()})
    dct.update({str(k.decode('utf-8')): np.array(v, dtype=np.float64) for k, v in dict(nfo_vecdbl).items()})
    dct.update({str(k.decode('utf-8')): np.array(v, dtype=int) for k, v in dict(nfo_vecint).items()})
    dct['root_indices'] = root_indices
    if root_out is not None:
        dct['root_out'] = root_out
    dct['mode'] = mode
    dct['success'] = success
    if nreached is not None:
        dct['nreached'] = nreached
    return dct


def integrate_adaptive(floating [:, ::1] y0,
                       floating [::1] x0,
                       floating [::1] xend,
                       floating [:, ::1] params,
                       atol,
                       realtype rtol,
                       dx0,
                       dx_min=None,
                       dx_max=None,
                       long int mxsteps=0,
                       str iter_type='undecided', str linear_solver="default", str method='BDF',
                       bool with_jacobian=True, bool return_on_root=False,
                       int autorestart=0, bool return_on_error=False, bool with_jtimes=False,
                       bool record_rhs_xvals=False, bool record_jac_xvals=False,
                       bool record_order=False, bool record_fpe=False,
                       realtype get_dx_max_factor=0.0, bool error_outside_bounds=False,
                       realtype max_invariant_violation=0.0, special_settings=None,
                       bool autonomous_exprs=False, int nprealloc=500, vector[realtype] constraints=[]):
    cdef:
        realtype ** xyout = <realtype **>malloc(y0.shape[0]*sizeof(realtype *))
        realtype [:,::1] xyout_view
        int * td = <int *>malloc(y0.shape[0]*sizeof(int))
        cnp.npy_intp dims[2]
        vector[CvodesOdeSys *] systems
        vector[vector[int]] root_indices
        list nfos = []
        string _lmm = method.lower().encode('UTF-8')
        string _iter_t = iter_type.lower().encode('UTF-8')
        vector[pair[int, vector[int]]] result
        int maxl=0
        realtype eps_lin=0.0
        unsigned nderiv=0,
        realtype [::1] _dx0
        realtype [::1] _dx_min
        realtype [::1] _dx_max
        bool success
        # this process is necessary since maybe the input type floating != realtype
        cnp.ndarray[realtype, ndim=1, mode='c'] xend_arr = np.asarray(xend, dtype=dtype)
        cnp.ndarray[realtype, ndim=2, mode='c'] params_arr = np.asarray(params, dtype=dtype)
        vector[realtype] atol_vec
        vector[realtype] special_settings_vec
        int idx, yi, tidx = 0
        realtype ** ew_ele = NULL

    if np.isnan(y0).any():
        raise ValueError("NaN found in y0")

    if isinstance(atol, Iterable):
        for at in atol:
            atol_vec.push_back(<realtype> at)
    else:
        atol_vec.push_back(<realtype> atol)
        atol_vec.resize(y0.shape[y0.ndim-1], atol[0])

    if special_settings is None:
        special_settings = []
    for ss in special_settings:
        special_settings_vec.push_back(<realtype> ss)

    if dx0 is None:
        _dx0 = np.zeros(y0.shape[0], dtype=dtype)
    else:
        _dx0 = np.ascontiguousarray(dx0, dtype=dtype)
        if _dx0.size == 1:
            _dx0 = _dx0*np.ones(y0.shape[0], dtype=dtype)
    if _dx0.size < y0.shape[0]:
        raise ValueError('dx0 too short')

    if dx_min is None:
        _dx_min = np.zeros(y0.shape[0], dtype=dtype)
    else:
        _dx_min = np.ascontiguousarray(dx_min, dtype=dtype)
        if _dx_min.size == 1:
            _dx_min = _dx_min*np.ones(y0.shape[0], dtype=dtype)
    if _dx_min.size < y0.shape[0]:
        raise ValueError('dx_min too short')

    if dx_max is None:
        _dx_max = np.zeros(y0.shape[0], dtype=dtype)
    else:
        _dx_max = np.ascontiguousarray(dx_max, dtype=dtype)
        if _dx_max.size == 1:
            _dx_max = _dx_max*np.ones(y0.shape[0], dtype=dtype)
    if _dx_max.size < y0.shape[0]:
        raise ValueError('dx_max too short')

    for idx in range(y0.shape[0]):
        systems.push_back(new CvodesOdeSys(<realtype *>(NULL) if params.shape[1] == 0
                                           else &params_arr[idx, 0], atol_vec, rtol,
                                           get_dx_max_factor, error_outside_bounds,
                                           max_invariant_violation, special_settings_vec))
        systems[idx].autonomous_exprs = autonomous_exprs
        systems[idx].record_rhs_xvals = record_rhs_xvals
        systems[idx].record_jac_xvals = record_jac_xvals
        systems[idx].record_order = record_order
        systems[idx].record_fpe = record_fpe
        td[idx] = nprealloc
        xyout[idx] = <realtype *>malloc(nprealloc*(y0.shape[1]+1)*sizeof(realtype))
        xyout[idx][0] = <realtype> x0[idx]
        for yi in range(y0.shape[1]):
            xyout[idx][yi+1] = <realtype> y0[idx, yi]

    try:
        result = multi_adaptive[CvodesOdeSys](
            xyout, td,
            systems, atol_vec, rtol, lmm_from_name(_lmm), &xend_arr[0], mxsteps,
            &_dx0[0], &_dx_min[0], &_dx_max[0], with_jacobian, iter_type_from_name(_iter_t),
            linear_solver_from_name(linear_solver.lower().encode('UTF-8')),
            maxl, eps_lin, nderiv, return_on_root, autorestart, return_on_error, with_jtimes,
            tidx, ew_ele, constraints
        )
        xout, yout = [], []
        for idx in range(y0.shape[0]):
            dims[0] = result[idx].first + 1
            dims[1] = y0.shape[1] + 1
            xyout_view = <realtype [:dims[0],:dims[1]:1]> xyout[idx]
            xyout_arr = np.asarray(xyout_view, dtype=dtype)
            xout.append(xyout_arr[:, 0])
            yout.append(xyout_arr[:, 1:])
            root_indices.push_back(result[idx].second)
            if return_on_error:
                if return_on_root and result[idx].second.size() > 0:
                    success = result[idx].second[result[idx].second.size() - 1] == result[idx].first
                else:
                    success = xout[-1][-1] == xend[idx]
            else:
                success = True

            nfos.append(_as_dict(systems[idx].current_info.nfo_int,
                                 systems[idx].current_info.nfo_dbl,
                                 systems[idx].current_info.nfo_vecdbl,
                                 systems[idx].current_info.nfo_vecint,
                                 root_indices[idx], root_out=None, mode='adaptive',
                                 success=success))

            del systems[idx]
    finally:
        free(td)
        # memory of xyout[i] is freed by xyout_arr taking ownership
        # but memory of xyout itself must be freed here
        free(xyout)

    return xout, yout, nfos


def integrate_predefined(floating [:, ::1] y0,
                         floating [:, ::1] xout,
                         floating [:, ::1] params,
                         atol,
                         realtype rtol,
                         dx0,
                         dx_min=None,
                         dx_max=None,
                         long int mxsteps=0,
                         str iter_type='undecided', str linear_solver="default", str method='BDF',
                         bool with_jacobian=True, int autorestart=0, bool return_on_error=False,
                         bool with_jtimes=False,
                         bool record_rhs_xvals=False, bool record_jac_xvals=False,
                         bool record_order=False, bool record_fpe=False,
                         realtype get_dx_max_factor=0.0, bool error_outside_bounds=False,
                         realtype max_invariant_violation=0.0, special_settings=None,
                         bool autonomous_exprs=False, vector[realtype] constraints=[]):
    cdef:
        vector[CvodesOdeSys *] systems
        list nfos = []
        cnp.ndarray[realtype, ndim=3, mode='c'] yout_arr
        string _lmm = method.lower().encode('UTF-8')
        string _iter_t = iter_type.lower().encode('UTF-8')
        vector[pair[int, pair[vector[int], vector[realtype]]]] result
        realtype [::1] _dx0
        realtype [::1] _dx_min
        realtype [::1] _dx_max
        int maxl = 0
        realtype eps_lin = 0.0
        unsigned nderiv = 0
        int nreached
        bool success
        cnp.ndarray[realtype, ndim=2, mode='c'] y0_arr = np.asarray(y0, dtype=dtype)
        cnp.ndarray[realtype, ndim=2, mode='c'] xout_arr = np.asarray(xout, dtype=dtype)
        cnp.ndarray[realtype, ndim=2, mode='c'] params_arr = np.asarray(params, dtype=dtype)
        vector[realtype] atol_vec
        vector[realtype] special_settings_vec
        realtype **ew_ele = NULL
    if np.isnan(y0).any():
        raise ValueError("NaN found in y0")

    if isinstance(atol, Iterable):
        for at in atol:
            atol_vec.push_back(<realtype> at)
    else:
        atol_vec.push_back(<realtype> atol)

    if special_settings is None:
        special_settings = []
    for ss in special_settings:
        special_settings_vec.push_back(<realtype> ss)

    if dx0 is None:
        _dx0 = np.zeros(y0.shape[0], dtype=dtype)
    else:
        _dx0 = np.ascontiguousarray(dx0, dtype=dtype)
        if _dx0.size == 1:
            _dx0 = _dx0*np.ones(y0.shape[0], dtype=dtype)
    if _dx0.size < y0.shape[0]:
        raise ValueError('dx0 too short')

    if dx_min is None:
        _dx_min = np.zeros(y0.shape[0], dtype=dtype)
    else:
        _dx_min = np.ascontiguousarray(dx_min, dtype=dtype)
        if _dx_min.size == 1:
            _dx_min = _dx_min*np.ones(y0.shape[0], dtype=dtype)
    if _dx_min.size < y0.shape[0]:
        raise ValueError('dx_min too short')

    if dx_max is None:
        _dx_max = np.zeros(y0.shape[0], dtype=dtype)
    else:
        _dx_max = np.ascontiguousarray(dx_max, dtype=dtype)
        if _dx_max.size == 1:
            _dx_max = _dx_max*np.ones(y0.shape[0], dtype=dtype)
    if _dx_max.size < y0.shape[0]:
        raise ValueError('dx_max too short')

    for idx in range(y0.shape[0]):
        systems.push_back(new CvodesOdeSys(<realtype *>(NULL) if params.shape[1] == 0
                                           else &params_arr[idx, 0], atol_vec, rtol,
                                           get_dx_max_factor, error_outside_bounds,
                                           max_invariant_violation, special_settings_vec))
        systems[idx].autonomous_exprs = autonomous_exprs
        systems[idx].record_rhs_xvals = record_rhs_xvals
        systems[idx].record_jac_xvals = record_jac_xvals
        systems[idx].record_order = record_order
        systems[idx].record_fpe = record_fpe


    yout_arr = np.empty((y0.shape[0], xout.shape[1], y0.shape[1]), dtype=dtype)
    result = multi_predefined[CvodesOdeSys](
        systems, atol, rtol, lmm_from_name(_lmm), <realtype *> y0_arr.data, xout.shape[1],
        <realtype *> xout_arr.data, <realtype *> yout_arr.data, mxsteps, &_dx0[0], &_dx_min[0],
        &_dx_max[0], with_jacobian, iter_type_from_name(_iter_t),
        linear_solver_from_name(linear_solver.lower().encode('UTF-8')),
        maxl, eps_lin, nderiv, autorestart, return_on_error, with_jtimes, ew_ele, constraints)

    for idx in range(y0.shape[0]):
        nreached = result[idx].first
        success = False if return_on_error and nreached < xout.shape[1] else True
        nfos.append(_as_dict(systems[idx].current_info.nfo_int,
                             systems[idx].current_info.nfo_dbl,
                             systems[idx].current_info.nfo_vecdbl,
                             systems[idx].current_info.nfo_vecint,
                             root_indices=result[idx].second.first,
                             root_out=result[idx].second.second, mode='predefined',
                             success=success, nreached=nreached))
        del systems[idx]

    return yout_arr, nfos
