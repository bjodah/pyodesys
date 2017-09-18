# -*- mode: cython -*-
# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: extra_compile_args = -std=c++14 -fopenmp
# distutils: extra_link_args = -fopenmp


from libc.stdlib cimport malloc, free
from libcpp cimport bool
from libcpp.utility cimport pair
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libcpp.string cimport string

cimport numpy as cnp
from odesys_anyode_iterative cimport OdeSys
from cvodes_cxx cimport lmm_from_name, iter_type_from_name
from cvodes_anyode_parallel cimport multi_predefined, multi_adaptive

import numpy as np

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(cnp.ndarray arr, int flags)

cnp.import_array()  # Numpy C-API initialization


from odesys_util cimport adaptive_return


cdef dict _as_dict(unordered_map[string, int] nfo,
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


def integrate_adaptive(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] y0,
                       cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] x0,
                       cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] xend,
                       cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] params,
                       vector[double] atol,
                       double rtol,
                       dx0,
                       dx_min=None,
                       dx_max=None,
                       long int mxsteps=0,
                       str iter_type='undecided', int linear_solver=0, str method='BDF',
                       bool with_jacobian=True, bool return_on_root=False,
                       int autorestart=0, bool return_on_error=False, bool with_jtimes=False,
                       bool record_rhs_xvals=False, bool record_jac_xvals=False,
                       bool record_order=False, bool record_fpe=False,
                       double get_dx_max_factor=-1.0, bool error_outside_bounds=False,
                       double max_invariant_violation=0.0, vector[double] special_settings=[],
                       bool autonomous_exprs=False):
    cdef:
        double ** xyout_arr = <double **>malloc(y0.shape[0]*sizeof(double*))
        int * td_arr = <int *>malloc(y0.shape[0]*sizeof(int))
        cnp.npy_intp dims[2]
        vector[OdeSys *] systems
        vector[vector[int]] root_indices
        list nfos = []
        string _lmm = method.lower().encode('UTF-8')
        string _iter_t = iter_type.lower().encode('UTF-8')
        vector[pair[int, vector[int]]] result
        int maxl=0
        double eps_lin=0.0
        unsigned nderiv=0,
        cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] _dx0
        cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] _dx_min
        cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] _dx_max
        bool success

    if np.isnan(y0).any():
        raise ValueError("NaN found in y0")

    if atol.size() == 1:
        atol.resize(y0.shape[y0.ndim-1], atol[0])

    if dx0 is None:
        _dx0 = np.zeros(y0.shape[0])
    else:
        _dx0 = np.ascontiguousarray(dx0, dtype=np.float64)
        if _dx0.size == 1:
            _dx0 = _dx0*np.ones(y0.shape[0])
    if _dx0.size < y0.shape[0]:
        raise ValueError('dx0 too short')

    if dx_min is None:
        _dx_min = np.zeros(y0.shape[0])
    else:
        _dx_min = np.ascontiguousarray(dx_min, dtype=np.float64)
        if _dx_min.size == 1:
            _dx_min = _dx_min*np.ones(y0.shape[0])
    if _dx_min.size < y0.shape[0]:
        raise ValueError('dx_min too short')

    if dx_max is None:
        _dx_max = np.zeros(y0.shape[0])
    else:
        _dx_max = np.ascontiguousarray(dx_max, dtype=np.float64)
        if _dx_max.size == 1:
            _dx_max = _dx_max*np.ones(y0.shape[0])
    if _dx_max.size < y0.shape[0]:
        raise ValueError('dx_max too short')

    for idx in range(y0.shape[0]):
        systems.push_back(new OdeSys(<double *>(NULL) if params.shape[1] == 0 else &params[idx, 0],
                                     atol, rtol, get_dx_max_factor, error_outside_bounds,
                                     max_invariant_violation, special_settings))
        systems[idx].autonomous_exprs = autonomous_exprs
        systems[idx].record_rhs_xvals = record_rhs_xvals
        systems[idx].record_jac_xvals = record_jac_xvals
        systems[idx].record_order = record_order
        systems[idx].record_fpe = record_fpe
        td_arr[idx] = 1
        xyout_arr[idx] = <double *>malloc((y0.shape[1]+1)*sizeof(double))
        xyout_arr[idx][0] = x0[idx]
        for yi in range(y0.shape[1]):
            xyout_arr[idx][yi+1] = y0[idx, yi]

    try:
        result = multi_adaptive[OdeSys](
            xyout_arr, td_arr,
            systems, atol, rtol, lmm_from_name(_lmm), <double *>xend.data, mxsteps,
            &_dx0[0], &_dx_min[0], &_dx_max[0], with_jacobian, iter_type_from_name(_iter_t), linear_solver,
            maxl, eps_lin, nderiv, return_on_root, autorestart, return_on_error, with_jtimes
        )
        xout, yout = [], []
        for idx in range(y0.shape[0]):
            dims[0] = result[idx].first + 1
            dims[1] = y0.shape[1] + 1
            xyout_np = cnp.PyArray_SimpleNewFromData(2, dims, cnp.NPY_DOUBLE, <void *>xyout_arr[idx])
            PyArray_ENABLEFLAGS(xyout_np, cnp.NPY_OWNDATA)
            xout.append(xyout_np[:, 0])
            yout.append(xyout_np[:, 1:])
            root_indices.push_back(result[idx].second)
            if return_on_error:
                if return_on_root and result[idx].second[result[idx].second.size() - 1] == result[idx].first:
                    success = True
                else:
                    success = xout[-1][-1] == xend[idx]
            else:
                success = True

            nfos.append(_as_dict(systems[idx].last_integration_info,
                                 systems[idx].last_integration_info_dbl,
                                 systems[idx].last_integration_info_vecdbl,
                                 systems[idx].last_integration_info_vecint,
                                 root_indices[idx], root_out=None, mode='adaptive',
                                 success=success))

            del systems[idx]
    finally:
        free(td_arr)

    return xout, yout, nfos


def integrate_predefined(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] y0,
                         cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] xout,
                         cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] params,
                         vector[double] atol,
                         double rtol,
                         dx0,
                         dx_min=None,
                         dx_max=None,
                         long int mxsteps=0,
                         str iter_type='undecided', int linear_solver=0, str method='BDF',
                         bool with_jacobian=True, int autorestart=0, bool return_on_error=False,
                         bool with_jtimes=False,
                         bool record_rhs_xvals=False, bool record_jac_xvals=False,
                         bool record_order=False, bool record_fpe=False,
                         double get_dx_max_factor=0.0, bool error_outside_bounds=False,
                         double max_invariant_violation=0.0, vector[double] special_settings=[],
                         bool autonomous_exprs=False):
    cdef:
        vector[OdeSys *] systems
        vector[vector[int]] root_indices
        vector[vector[double]] root_out
        list nfos = []
        cnp.ndarray[cnp.float64_t, ndim=3, mode='c'] yout
        string _lmm = method.lower().encode('UTF-8')
        string _iter_t = iter_type.lower().encode('UTF-8')
        vector[pair[int, pair[vector[int], vector[double]]]] result
        cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] _dx0
        cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] _dx_min
        cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] _dx_max
        int maxl = 0
        double eps_lin = 0.0
        unsigned nderiv = 0
        int nreached
        bool success

    if np.isnan(y0).any():
        raise ValueError("NaN found in y0")

    if atol.size() == 1:
        atol.resize(y0.shape[y0.ndim-1], atol[0])

    if dx0 is None:
        _dx0 = np.zeros(y0.shape[0])
    else:
        _dx0 = np.ascontiguousarray(dx0, dtype=np.float64)
        if _dx0.size == 1:
            _dx0 = _dx0*np.ones(y0.shape[0])
    if _dx0.size < y0.shape[0]:
        raise ValueError('dx0 too short')

    if dx_min is None:
        _dx_min = np.zeros(y0.shape[0])
    else:
        _dx_min = np.ascontiguousarray(dx_min, dtype=np.float64)
        if _dx_min.size == 1:
            _dx_min = _dx_min*np.ones(y0.shape[0])
    if _dx_min.size < y0.shape[0]:
        raise ValueError('dx_min too short')

    if dx_max is None:
        _dx_max = np.zeros(y0.shape[0])
    else:
        _dx_max = np.ascontiguousarray(dx_max, dtype=np.float64)
        if _dx_max.size == 1:
            _dx_max = _dx_max*np.ones(y0.shape[0])
    if _dx_max.size < y0.shape[0]:
        raise ValueError('dx_max too short')

    for idx in range(y0.shape[0]):
        systems.push_back(new OdeSys(<double *>(NULL) if params.shape[1] == 0 else &params[idx, 0],
                                     atol, rtol, get_dx_max_factor, error_outside_bounds,
                                     max_invariant_violation, special_settings))
        systems[idx].autonomous_exprs = autonomous_exprs
        systems[idx].record_rhs_xvals = record_rhs_xvals
        systems[idx].record_jac_xvals = record_jac_xvals
        systems[idx].record_order = record_order
        systems[idx].record_fpe = record_fpe


    yout = np.empty((y0.shape[0], xout.shape[1], y0.shape[1]))
    result = multi_predefined[OdeSys](
        systems, atol, rtol, lmm_from_name(_lmm), <double *>y0.data, xout.shape[1], <double *>xout.data,
        <double *>yout.data, mxsteps, &_dx0[0], &_dx_min[0], &_dx_max[0], with_jacobian,
        iter_type_from_name(_iter_t), linear_solver, maxl, eps_lin, nderiv, autorestart,
        return_on_error, with_jtimes)

    for idx in range(y0.shape[0]):
        nreached = result[idx].first
        root_indices.push_back(result[idx].second.first)
        root_out.push_back(result[idx].second.second)
        success = False if return_on_error and nreached < xout.shape[1] else True
        nfos.append(_as_dict(systems[idx].last_integration_info,
                             systems[idx].last_integration_info_dbl,
                             systems[idx].last_integration_info_vecdbl,
                             systems[idx].last_integration_info_vecint,
                             root_indices, root_out=root_out, mode='predefined',
                             success=success, nreached=nreached))
        del systems[idx]

    yout_arr = np.asarray(yout)
    return yout, nfos
