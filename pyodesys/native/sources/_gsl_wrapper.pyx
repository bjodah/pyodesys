# -*- mode: cython -*-
# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: extra_compile_args = -std=c++11

from libcpp cimport bool
from libcpp.vector cimport vector

cimport numpy as cnp
from odesys_anyode cimport OdeSys
from gsl_odeiv2_cxx cimport styp_from_name
from gsl_odeiv2_anyode_nogil cimport simple_predefined, simple_adaptive

import numpy as np


cdef dict get_last_info(OdeSys * odesys):
    info = {str(k.decode('utf-8')): v for k, v in dict(odesys.last_integration_info).items()}
    info['nfev'] = odesys.m_nfev
    info['njev'] = odesys.m_njev
    return info


def integrate_predefined(cnp.ndarray[cnp.float64_t, ndim=1] y0,
                         cnp.ndarray[cnp.float64_t, ndim=1] xout,
                         cnp.ndarray[cnp.float64_t, ndim=1] params,
                         double atol,
                         double rtol,
                         double dx0=0.0, double dx_min=0.0, double dx_max=0.0, long int mxsteps=0,
                         str method='bsimp'):
    cdef:
        OdeSys * odesys
        cnp.ndarray[cnp.float64_t, ndim=2] yout

    if np.isnan(y0).any():
        raise ValueError("NaN found in y0")
    odesys = new OdeSys(<double *>(NULL) if params.size == 0 else &params[0])
    try:
        yout = np.empty((xout.size, odesys.get_ny()))
        simple_predefined[OdeSys](odesys, atol, rtol, styp_from_name(method.upper().encode('UTF-8')),
                                  &y0[0], xout.size, &xout[0], &yout[0, 0], mxsteps,
                                  dx0, dx_min, dx_max)
        return yout, get_last_info(odesys)
    finally:
        del odesys
        del err_handler

def integrate_adaptive(cnp.ndarray[cnp.float64_t, ndim=1] y0,
                       double x0, double xend,
                       cnp.ndarray[cnp.float64_t, ndim=1] params,
                       double atol,
                       double rtol,
                       double dx0=0.0, double dx_min=0.0, double dx_max=0.0, long int mxsteps=0,
                       str method='bsimp'):
    cdef:
        OdeSys * odesys
        size_t nsteps

    if np.isnan(y0).any():
        raise ValueError("NaN found in y0")
    odesys = new OdeSys(<double *>(NULL) if params.size == 0 else &params[0])
    try:
        xout, yout = simple_adaptive[OdeSys](
            odesys, atol, rtol, styp_from_name(method.upper().encode('UTF-8')),
            &y0[0], x0, xend, mxsteps, dx0, dx_min, dx_max)
        yout = np.asarray(yout)
        return np.asarray(xout), yout.reshape((len(xout), odesys.get_ny())), get_last_info(odesys)
    finally:
        del odesys
        del err_handler
