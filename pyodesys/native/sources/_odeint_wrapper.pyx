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
from odeint_anyode cimport styp_from_name
from odeint_anyode_parallel cimport multi_predefined, multi_adaptive

import numpy as np


cdef list _as_dict(vector[unordered_map[string, int]] nfos, mode=None):
    py_nfos = []
    for idx in range(nfos.size()):
        dct = {str(k.decode('utf-8')): v for k, v in dict(nfos[idx]).items()}
        dct['mode'] = mode
        py_nfos.append(dct)
    return py_nfos


def integrate_adaptive(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] y0,
                       cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] x0,
                       cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] xend,
                       cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] params,
                       double atol, double rtol,
                       double dx0=0.0,
                       long int mxsteps=0, str method='rosenbrock4'):
    cdef:
        vector[OdeSys *] systems
        vector[unordered_map[string, int]] nfos
        string _styp = method.lower().encode('UTF-8')
        vector[pair[vector[double], vector[double]]] result

    if np.isnan(y0).any():
        raise ValueError("NaN found in y0")

    for idx in range(y0.shape[0]):
        systems.push_back(new OdeSys(<double *>(NULL) if params.shape[1] == 0 else &params[idx, 0]))

    result = multi_adaptive[OdeSys](
        systems, atol, rtol, styp_from_name(_styp), <double *>y0.data,
        <double *>x0.data, <double *>xend.data, mxsteps,
        dx0)

    xout, yout = [], []
    for idx in range(y0.shape[0]):
        _xout = np.asarray(result[idx].first)
        xout.append(_xout)
        _yout = np.asarray(result[idx].second)
        yout.append(_yout.reshape((_xout.size, y0.shape[1])))
        nfos.push_back(systems[idx].last_integration_info)
        del systems[idx]

    yout_arr = [np.asarray(_) for _ in yout]

    return (xout, yout, _as_dict(nfos, mode='adaptive'))


def integrate_predefined(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] y0,
                         cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] xout,
                         cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] params,
                         double atol, double rtol,
                         double dx0=0.0,
                         long int mxsteps=0, str method='rosenbrock4'):
    cdef:
        vector[OdeSys *] systems
        vector[unordered_map[string, int]] nfos
        cnp.ndarray[cnp.float64_t, ndim=3, mode='c'] yout
        string _styp = method.lower().encode('UTF-8')

    if np.isnan(y0).any():
        raise ValueError("NaN found in y0")

    for idx in range(y0.shape[0]):
        systems.push_back(new OdeSys(<double *>(NULL) if params.shape[1] == 0 else &params[idx, 0]))

    yout = np.empty((y0.shape[0], xout.shape[1], y0.shape[1]))
    multi_predefined[OdeSys](
        systems, atol, rtol, styp_from_name(_styp), <double *>y0.data, xout.shape[1],
        <double *>xout.data, <double *>yout.data,
        mxsteps, dx0)

    for idx in range(y0.shape[0]):
        nfos.push_back(systems[idx].last_integration_info)
        del systems[idx]

    yout_arr = np.asarray(yout)
    return (
        yout,
        _as_dict(nfos, mode='predefined')
    )
