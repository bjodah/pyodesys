# -*- mode: cython -*-
# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: extra_compile_args = -std=c++11 -fopenmp
# distutils: extra_link_args = -fopenmp


from libcpp cimport bool
from libcpp.utility cimport pair
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libcpp.string cimport string

cimport numpy as cnp
from odesys_anyode cimport OdeSys
from odesys_util cimport adaptive_return
from cvodes_cxx cimport lmm_from_name, iter_type_from_name
from cvodes_anyode_parallel cimport multi_predefined, multi_adaptive

import numpy as np


cdef list _as_dict(vector[unordered_map[string, int]] nfos,
                   root_indices, root_out=None, mode=None):
    py_nfos = []
    for idx in range(nfos.size()):
        dct = {str(k.decode('utf-8')): v for k, v in dict(nfos[idx]).items()}
        dct['root_indices'] = root_indices[idx]
        if root_out is not None:
            dct['root_out'] = root_out[idx]
        dct['mode'] = mode
        py_nfos.append(dct)
    return py_nfos


def integrate_adaptive(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] y0,
                       cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] x0,
                       cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] xend,
                       cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] params,
                       vector[double] atol,
                       double rtol,
                       double dx0=0.0, double dx_min=0.0, double dx_max=0.0,
                       long int mxsteps=0,
                       str iter_type='undecided', int linear_solver=0, str method='BDF',
                       bool with_jacobian=True):
    cdef:
        vector[OdeSys *] systems
        vector[vector[int]] root_indices
        vector[unordered_map[string, int]] nfos
        string _lmm = method.lower().encode('UTF-8')
        string _iter_t = iter_type.lower().encode('UTF-8')
        vector[pair[pair[vector[double], vector[double]], vector[int]]] result

    if np.isnan(y0).any():
        raise ValueError("NaN found in y0")

    for idx in range(y0.shape[0]):
        systems.push_back(new OdeSys(<double *>(NULL) if params.shape[1] == 0 else &params[idx, 0]))

    result = multi_adaptive[OdeSys](
        systems, atol, rtol, lmm_from_name(_lmm), <double *>y0.data,
        <double *>x0.data, <double *>xend.data, mxsteps,
        dx0, dx_min, dx_max, with_jacobian, iter_type_from_name(_iter_t), linear_solver)

    xout, yout = [], []
    for idx in range(y0.shape[0]):
        _xout = np.asarray(result[idx].first.first)
        xout.append(_xout)
        _yout = np.asarray(result[idx].first.second)
        yout.append(_yout.reshape((_xout.size, y0.shape[1])))
        root_indices.push_back(result[idx].second)
        nfos.push_back(systems[idx].last_integration_info)
        del systems[idx]

    yout_arr = [np.asarray(_) for _ in yout]

    return (xout, yout, _as_dict(nfos, root_indices, root_out=None, mode='adaptive'))


def integrate_predefined(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] y0,
                         cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] xout,
                         cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] params,
                         vector[double] atol,
                         double rtol,
                         double dx0=0.0, double dx_min=0.0, double dx_max=0.0,
                         long int mxsteps=0,
                         str iter_type='undecided', int linear_solver=0, str method='BDF',
                         bool with_jacobian=True):
    cdef:
        vector[OdeSys *] systems
        vector[vector[int]] root_indices
        vector[vector[double]] root_out
        vector[unordered_map[string, int]] nfos
        cnp.ndarray[cnp.float64_t, ndim=3, mode='c'] yout
        string _lmm = method.lower().encode('UTF-8')
        string _iter_t = iter_type.lower().encode('UTF-8')
        vector[pair[vector[int], vector[double]]] result

    if np.isnan(y0).any():
        raise ValueError("NaN found in y0")

    for idx in range(y0.shape[0]):
        systems.push_back(new OdeSys(<double *>(NULL) if params.shape[1] == 0 else &params[idx, 0]))

    yout = np.empty((y0.shape[0], xout.shape[1], y0.shape[1]))
    result = multi_predefined[OdeSys](
        systems, atol, rtol, lmm_from_name(_lmm), <double *>y0.data, xout.shape[1], <double *>xout.data, <double *>yout.data,
        mxsteps, dx0, dx_min, dx_max, with_jacobian, iter_type_from_name(_iter_t), linear_solver)

    for idx in range(y0.shape[0]):
        root_indices.push_back(result[idx].first)
        root_out.push_back(result[idx].second)
        nfos.push_back(systems[idx].last_integration_info)
        del systems[idx]

    yout_arr = np.asarray(yout)
    return (
        yout,
        _as_dict(nfos, root_indices, root_out, mode='predefined')
    )
