// ${'\n// '.join(_message_for_rendered)}
// -*- coding: utf-8 -*-
<%doc>
This is file is a mako-formatted template
</%doc>
// User provided system description: ${p_odesys.description}
// Names of dependent variables: ${p_odesys.names}
// Names of parameters: ${p_odesys.param_names}

#include <algorithm>
#include <iostream>
#include <limits>
#include <math.h>
#include <vector>
%for inc in p_includes:
#include ${inc}
%endfor

%if p_anon is not None:
namespace {  // anonymous namespace for user-defined helper functions
    std::vector<std::string> p_odesys_names ${'' if p_odesys.names is None else '= {"%s"}' % '", "'.join(p_odesys.names)};
    ${p_anon}
}
%endif
using odesys_anyode::OdeSys;

OdeSys::OdeSys(const double * const params, std::vector<double> atol, double rtol,
               double get_dx_max_factor, bool error_outside_bounds) :
    m_p_cse(${p_common['nsubs']}), m_atol(atol), m_rtol(rtol), m_get_dx_max_factor(get_dx_max_factor),
    m_error_outside_bounds(error_outside_bounds) {
    m_p.assign(params, params + ${len(p_odesys.params)});
    <% idx = 0 %>
  % for cse_token, cse_expr in p_common['cses']:
   %if cse_token.startswith('m_p_cse'):
    ${cse_token} = ${cse_expr}; <% assert cse_token == 'm_p_cse[{0}]'.format(idx); idx += 1 %>
   %else:
    const auto ${cse_token} = ${cse_expr};
   %endif
  % endfor
    use_get_dx_max = (m_get_dx_max_factor > 0.0) ? ${'true' if p_get_dx_max else 'false'} : false;
}
int OdeSys::get_ny() const {
    return ${p_odesys.ny};
}
int OdeSys::get_nroots() const {
    return ${p_odesys.nroots};
}
AnyODE::Status OdeSys::rhs(double x,
                           const double * const __restrict__ y,
                           double * const __restrict__ f) {
    ${'AnyODE::ignore(x);' if p_odesys.autonomous_exprs else ''}
  % for cse_token, cse_expr in p_rhs['cses']:
    const auto ${cse_token} = ${cse_expr};
  % endfor

  % for i, expr in enumerate(p_rhs['exprs']):
    f[${i}] = ${expr};
  % endfor
    this->nfev++;
  % if p_support_recoverable_error:
    if (m_error_outside_bounds){
        if (m_lower_bounds.size() > 0) {
            for (int i=0; i < ${p_odesys.ny}; ++i) {
                if (y[i] < m_lower_bounds[i]) {
                    std::cerr << "Lower bound (" << m_lower_bounds[0] << ") for "
                              << (p_odesys_names.size() ? p_odesys_names[i] : std::to_string(i))
                              << " exceeded (" << y[i] << ") at x="<< x << "\n";
                    return AnyODE::Status::recoverable_error;
                }
            }
        }
        if (m_upper_bounds.size() > 0) {
            for (int i=0; i < ${p_odesys.ny}; ++i) {
                if (y[i] > m_upper_bounds[i]) {
                    std::cerr << "Upper bound (" << m_upper_bounds[0] << ") for "
                              << (p_odesys_names.size() ? p_odesys_names[i] : std::to_string(i))
                              << " exceeded (" << y[i] << ") at x="<< x << "\n";
                    return AnyODE::Status::recoverable_error;
                }
            }
        }
    }
  % endif
  % if getattr(p_odesys, '_nonnegative', False) and p_support_recoverable_error:
    for (int i=0; i<${p_odesys.ny}; ++i) if (y[i] < 0) return AnyODE::Status::recoverable_error;
  % endif
    return AnyODE::Status::success;
}

% if p_jac is not None:
% for order in ('cmaj', 'rmaj'):

AnyODE::Status OdeSys::dense_jac_${order}(double x,
                                      const double * const __restrict__ y,
                                      const double * const __restrict__ fy,
                                      double * const __restrict__ jac,
                                      long int ldim,
                                      double * const __restrict__ dfdt) {
    // The AnyODE::ignore(...) calls below are used to generate code free from false compiler warnings.
    AnyODE::ignore(fy);  // Currently we are not using fy (could be done through extensive pattern matching)
    ${'AnyODE::ignore(x);' if p_odesys.autonomous_exprs else ''}
    ${'AnyODE::ignore(y);' if (not any([yi in p_odesys.get_jac().free_symbols for yi in p_odesys.dep]) and
                               not any([yi in p_odesys.get_dfdx().free_symbols for yi in p_odesys.dep])) else ''}

  % for cse_token, cse_expr in p_jac['cses']:
    const auto ${cse_token} = ${cse_expr};
  % endfor

  % for i_major in range(p_odesys.ny):
   % for i_minor in range(p_odesys.ny):
<%
      curr_expr = p_jac['exprs'][i_minor, i_major] if order == 'cmaj' else p_jac['exprs'][i_major, i_minor]
      if curr_expr == '0' and p_jacobian_set_to_zero_by_solver:
          continue
%>    jac[ldim*${i_major} + ${i_minor}] = ${curr_expr};
   % endfor

  % endfor
    if (dfdt){
      % for idx, expr in enumerate(p_jac['dfdt_exprs']):
        dfdt[${idx}] = ${expr};
      % endfor
    }
    this->njev++;
    return AnyODE::Status::success;
}
% endfor
% endif

double OdeSys::get_dx0(double x, const double * const y) {
% if p_first_step is None:
    AnyODE::ignore(x); AnyODE::ignore(y);  // avoid compiler warning about unused parameter.
    return 0.0;  // invokes the default behaviour of the chosen solver
% elif isinstance(p_first_step, str):
    ${p_first_step}
% else:
  % for cse_token, cse_expr in p_first_step['cses']:
    const double ${cse_token} = ${cse_expr};
  % endfor
    ${'' if p_odesys.indep in p_odesys.first_step_expr.free_symbols else 'AnyODE::ignore(x);'}
    ${'' if any([yi in p_odesys.first_step_expr.free_symbols for yi in p_odesys.dep]) else 'AnyODE::ignore(y);'}
    return ${p_first_step['expr']};
% endif
}

double OdeSys::get_dx_max(double x, const double * const y) {
% if p_get_dx_max is False:
    AnyODE::ignore(x); AnyODE::ignore(y);  // avoid compiler warning about unused parameter.
    return INFINITY;
% elif p_get_dx_max is True:
    auto fvec = std::vector<double>(${p_odesys.ny});
    auto hvec = std::vector<double>(${p_odesys.ny});
    rhs(x, y, &fvec[0]);
    for (int idx=0; idx<${p_odesys.ny}; ++idx){
        if (fvec[idx] == 0) {
            hvec[idx] = std::numeric_limits<double>::infinity();
        } else if (fvec[idx] > 0) {
            hvec[idx] = std::abs(m_upper_bounds[idx] - y[idx])/fvec[idx];
        } else { // fvec[idx] < 0
            hvec[idx] = std::abs((m_lower_bounds[idx] - y[idx])/fvec[idx]);
        }
    }
    const auto result = *std::min_element(std::begin(hvec), std::end(hvec));
    if (m_get_dx_max_factor == 0.0)
        return result;
    else if (m_get_dx_max_factor < 0.0)
        return -m_get_dx_max_factor*result;
    else
        return m_get_dx_max_factor*result;
% elif isinstance(p_get_dx_max, str):
    ${p_get_dx_max}
% else:
    <% raise NotImplementedError("Don't know what to do with: {}".format(p_get_dx_max)) %>
% endif
}

AnyODE::Status OdeSys::roots(double x, const double * const y, double * const out) {
% if p_odesys.roots is None:
    AnyODE::ignore(x); AnyODE::ignore(y); AnyODE::ignore(out);
    return AnyODE::Status::success;
% else:
  % for cse_token, cse_expr in p_roots['cses']:
    const auto ${cse_token} = ${cse_expr};
  % endfor

  % for i, expr in enumerate(p_roots['exprs']):
    out[${i}] = ${expr};
  % endfor
    this->nrev++;
    return AnyODE::Status::success;
% endif
}
