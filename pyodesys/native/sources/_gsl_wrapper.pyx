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
from gsl_odeiv2_cxx cimport styp_from_name
from gsl_odeiv2_anyode_parallel cimport multi_predefined, multi_adaptive

import numpy as np


cdef dict _as_dict(unordered_map[string, int] nfo,
                   unordered_map[string, double] nfo_dbl,
                   success, mode=None, nreached=None):
    dct = {str(k.decode('utf-8')): v for k, v in dict(nfo).items()}
    dct.update({str(k.decode('utf-8')): v for k, v in dict(nfo_dbl).items()})
    dct['mode'] = mode
    dct['success'] = success
    if nreached is not None:
        dct['nreached'] = nreached
    return dct


def integrate_adaptive(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] y0,
                       cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] x0,
                       cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] xend,
                       cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] params,
                       double atol, double rtol,
                       dx0, dx_min=None, dx_max=None,
                       long int mxsteps=0, str method='bsimp', int autorestart=0,
                       bool return_on_error=False, double get_dx_max_factor=-1.0,
                       vector[double] special_settings=[]):
    cdef:
        vector[OdeSys *] systems
        list nfos = []
        string _styp = method.lower().encode('UTF-8')
        vector[pair[vector[double], vector[double]]] result
        cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] _dx0
        cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] _dx_min
        cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] _dx_max

    if np.isnan(y0).any():
        raise ValueError("NaN found in y0")

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
                                     [atol], rtol, get_dx_max_factor, False, 0.0, special_settings))

    result = multi_adaptive[OdeSys](
        systems, atol, rtol, styp_from_name(_styp), <double *>y0.data,
        <double *>x0.data, <double *>xend.data, mxsteps,
        &_dx0[0], &_dx_min[0], &_dx_max[0], autorestart, return_on_error)

    xout, yout = [], []
    for idx in range(y0.shape[0]):
        _xout = np.asarray(result[idx].first)
        xout.append(_xout)
        _yout = np.asarray(result[idx].second)
        yout.append(_yout.reshape((_xout.size, y0.shape[1])))
        nfos.append(_as_dict(systems[idx].current_info.nfo_int,
                             systems[idx].current_info.nfo_dbl,
                             xend[idx] == _xout[-1], mode='adaptive'))
        del systems[idx]

    yout_arr = [np.asarray(_) for _ in yout]

    return (xout, yout, nfos)


def integrate_predefined(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] y0,
                         cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] xout,
                         cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] params,
                         double atol, double rtol,
                         dx0, dx_min=None, dx_max=None,
                         long int mxsteps=0, str method='bsimp',
                         bool return_on_error=False,
                         double get_dx_max_factor=0.0,
                         vector[double] special_settings=[]):
    cdef:
        vector[OdeSys *] systems
        list nfos = []
        cnp.ndarray[cnp.float64_t, ndim=3, mode='c'] yout
        string _styp = method.lower().encode('UTF-8')
        cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] _dx0
        cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] _dx_min
        cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] _dx_max
        vector[int] result
        int nreached
        bool success

    if np.isnan(y0).any():
        raise ValueError("NaN found in y0")

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
                                     [atol], rtol, get_dx_max_factor, False, 0.0, special_settings))

    yout = np.empty((y0.shape[0], xout.shape[1], y0.shape[1]))
    result = multi_predefined[OdeSys](
        systems, atol, rtol, styp_from_name(_styp), <double *>y0.data, xout.shape[1],
        <double *>xout.data, <double *>yout.data,
        mxsteps, &_dx0[0], &_dx_min[0], &_dx_max[0])

    for idx in range(y0.shape[0]):
        nreached = result[idx]
        success = False if return_on_error and nreached < xout.shape[1] else True
        nfos.append(_as_dict(systems[idx].current_info.nfo_int,
                             systems[idx].current_info.nfo_dbl,
                             mode='predefined', success=success, nreached=nreached))
        del systems[idx]

    yout_arr = np.asarray(yout)
    return yout, nfos
