// -*- coding: utf-8 -*-
// -*- ${'mode: read-only'} -*-
// ${_message_for_rendered}

// User provided system description: ${p_odesys.description}
// Names of dependent variables: ${p_odesys.names}

#include <math.h>
#include <vector>
#include "odesys_anyode.hpp"

using odesys_anyode::OdeSys;

OdeSys::OdeSys(const double * const params) {
    m_p.assign(params, params + ${len(p_odesys.params)});
}
int OdeSys::get_ny() const {
    return ${p_odesys.ny};
}
AnyODE::Status OdeSys::rhs(double t,
                           const double * const __restrict__ y,
                           double * const __restrict__ f) {
    ${'AnyODE::ignore(t);' if p_odesys.autonomous else ''}
  % for cse_token, cse_expr in p_rhs_cses:
    const double ${cse_token} = ${cse_expr};
  % endfor

  % for i, expr in enumerate(p_rhs_exprs):
    f[${i}] = ${expr};
  % endfor
    this->nfev++;
    return AnyODE::Status::success;
}

% for order in ('cmaj', 'rmaj'):

AnyODE::Status OdeSys::dense_jac_${order}(double t,
                                      const double * const __restrict__ y,
                                      const double * const __restrict__ fy,
                                      double * const __restrict__ jac,
                                      long int ldim,
                                      double * const __restrict__ dfdt) {
    // The AnyODE::ignore(...) calls below are used to generate code free from compiler warnings.
    AnyODE::ignore(fy);  // Currently we are not using fy (could be done through extensive pattern matching)
    ${'AnyODE::ignore(t);' if p_odesys.autonomous else ''}
    ${'AnyODE::ignore(y);' if (not any([yi in p_odesys.get_jac().free_symbols for yi in p_odesys.dep]) and
                               not any([yi in p_odesys.get_dfdx().free_symbols for yi in p_odesys.dep])) else ''}

  % for cse_token, cse_expr in p_jac_cses:
    const double ${cse_token} = ${cse_expr};
  % endfor

  % for i_major in range(p_odesys.ny):
   % for i_minor in range(p_odesys.ny):
    jac[ldim*${i_major} + ${i_minor}] = ${p_jac_exprs[i_minor, i_major] if order == 'cmaj' else p_jac_exprs[i_major, i_minor]};
   % endfor

  % endfor
    if (dfdt){
      % for idx, expr in enumerate(p_jac_dfdt_exprs):
        dfdt[${idx}] = ${expr};
      % endfor
    }
    this->njev++;
    return AnyODE::Status::success;
}
% endfor
