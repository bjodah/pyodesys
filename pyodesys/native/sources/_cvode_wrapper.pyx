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
from cvodes_cxx cimport lmm_from_name, iter_type_from_name
from cvodes_anyode_parallel cimport multi_predefined, multi_adaptive

import numpy as np

from odesys_util cimport adaptive_return


cdef extern from "odesys_anyode_iterative.hpp" namespace "odesys_anyode":
    cdef cppclass OdeSys:
        OdeSys(const double * const) nogil except +
        unordered_map[string, int] last_integration_info
        unordered_map[string, double] last_integration_info_dbl


cdef dict _as_dict(unordered_map[string, int] nfo,
                   unordered_map[string, double] nfo_dbl,
                   root_indices, root_out=None, mode=None):
    dct = {str(k.decode('utf-8')): v for k, v in dict(nfo).items()}
    dct.update({str(k.decode('utf-8')): v for k, v in dict(nfo_dbl).items()})
    dct['root_indices'] = root_indices
    if root_out is not None:
        dct['root_out'] = root_out
    dct['mode'] = mode
    return dct


def integrate_adaptive(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] y0,
                       cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] x0,
                       cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] xend,
                       cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] params,
                       vector[double] atol,
                       double rtol,
                       cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] dx0,
                       cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] dx_min=None,
                       cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] dx_max=None,
                       long int mxsteps=0,
                       str iter_type='undecided', int linear_solver=0, str method='BDF',
                       bool with_jacobian=True, int autorestart=0, bool return_on_error=False):
    cdef:
        vector[OdeSys *] systems
        vector[vector[int]] root_indices
        list nfos = []
        string _lmm = method.lower().encode('UTF-8')
        string _iter_t = iter_type.lower().encode('UTF-8')
        vector[pair[pair[vector[double], vector[double]], vector[int]]] result
        int maxl=0
        double eps_lin=0.0
        unsigned nderiv=0,
        bool return_on_root=False

    if np.isnan(y0).any():
        raise ValueError("NaN found in y0")

    if dx0 is None:
        dx0 = np.zeros(y0.shape[0])
    if dx_min is None:
        dx_min = np.zeros(y0.shape[0])
    if dx_max is None:
        dx_max = np.zeros(y0.shape[0])

    if dx0.size < y0.shape[0]:
        raise ValueError('dx0 too short')
    if dx_min.size < y0.shape[0]:
        raise ValueError('dx_min too short')
    if dx_max.size < y0.shape[0]:
        raise ValueError('dx_max too short')

    for idx in range(y0.shape[0]):
        systems.push_back(new OdeSys(<double *>(NULL) if params.shape[1] == 0 else &params[idx, 0]))

    result = multi_adaptive[OdeSys](
        systems, atol, rtol, lmm_from_name(_lmm), <double *>y0.data,
        <double *>x0.data, <double *>xend.data, mxsteps,
        &dx0[0], &dx_min[0], &dx_max[0], with_jacobian, iter_type_from_name(_iter_t), linear_solver,
        maxl, eps_lin, nderiv, return_on_root, autorestart, return_on_error
    )

    xout, yout = [], []
    for idx in range(y0.shape[0]):
        _xout = np.asarray(result[idx].first.first)
        xout.append(_xout)
        _yout = np.asarray(result[idx].first.second)
        yout.append(_yout.reshape((_xout.size, y0.shape[1])))
        root_indices.push_back(result[idx].second)
        nfos.append(_as_dict(systems[idx].last_integration_info,
                             systems[idx].last_integration_info_dbl,
                             root_indices[idx], root_out=None, mode='adaptive'))
        del systems[idx]

    yout_arr = [np.asarray(_) for _ in yout]

    return xout, yout, nfos


def integrate_predefined(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] y0,
                         cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] xout,
                         cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] params,
                         vector[double] atol,
                         double rtol,
                         cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] dx0,
                         cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] dx_min=None,
                         cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] dx_max=None,
                         long int mxsteps=0,
                         str iter_type='undecided', int linear_solver=0, str method='BDF',
                         bool with_jacobian=True, int autorestart=0):
    cdef:
        vector[OdeSys *] systems
        vector[vector[int]] root_indices
        vector[vector[double]] root_out
        list nfos = []
        cnp.ndarray[cnp.float64_t, ndim=3, mode='c'] yout
        string _lmm = method.lower().encode('UTF-8')
        string _iter_t = iter_type.lower().encode('UTF-8')
        vector[pair[vector[int], vector[double]]] result

    if np.isnan(y0).any():
        raise ValueError("NaN found in y0")

    if dx0 is None:
        dx0 = np.zeros(y0.shape[0])
    if dx_min is None:
        dx_min = np.zeros(y0.shape[0])
    if dx_max is None:
        dx_max = np.zeros(y0.shape[0])

    if dx0.size < y0.shape[0]:
        raise ValueError('dx0 too short')
    if dx_min.size < y0.shape[0]:
        raise ValueError('dx_min too short')
    if dx_max.size < y0.shape[0]:
        raise ValueError('dx_max too short')

    for idx in range(y0.shape[0]):
        systems.push_back(new OdeSys(<double *>(NULL) if params.shape[1] == 0 else &params[idx, 0]))

    yout = np.empty((y0.shape[0], xout.shape[1], y0.shape[1]))
    result = multi_predefined[OdeSys](
        systems, atol, rtol, lmm_from_name(_lmm), <double *>y0.data, xout.shape[1], <double *>xout.data, <double *>yout.data,
        mxsteps, &dx0[0], &dx_min[0], &dx_max[0], with_jacobian, iter_type_from_name(_iter_t), linear_solver, autorestart)

    for idx in range(y0.shape[0]):
        root_indices.push_back(result[idx].first)
        root_out.push_back(result[idx].second)
        nfos.append(_as_dict(systems[idx].last_integration_info,
                             systems[idx].last_integration_info_dbl,
                             root_indices, root_out, mode='predefined'))
        del systems[idx]

    yout_arr = np.asarray(yout)
    return yout, nfos
