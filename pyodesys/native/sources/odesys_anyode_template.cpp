// ${'\n// '.join(_message_for_rendered)}
// -*- coding: utf-8 -*-
<%doc>
This is file is a mako template for a C++ source file defining the ODE system.
</%doc>
// User provided system description: ${p_odesys.description}
// Names of dependent variables: ${p_odesys.names}
// Names of parameters: ${p_odesys.param_names}

#include <algorithm>
#include <iostream>
#include <limits>
#include <cmath>
#include <vector>
%for inc in p_includes:
#include ${inc}
%endfor

namespace {  // anonymous namespace for user-defined helper functions
    std::vector<std::string> p_odesys_names ${'' if p_odesys.names is None else '= {"%s"}' % '", "'.join(p_odesys.names)};
%if p_anon is not None:
    ${p_anon}
%endif
}

using namespace std;

typedef ${p_realtype} realtype;
typedef ${p_indextype} indextype;

namespace odesys_anyode {
    template <>
    struct OdeSys<realtype, indextype>: public AnyODE::${p_baseclass}<realtype, indextype> {
        std::vector<realtype> m_p;
        std::vector<realtype> m_p_cse;
        std::vector<realtype> m_atol;
        std::vector<realtype> m_upper_bounds;
        std::vector<realtype> m_lower_bounds;
        std::vector<realtype> m_invar0;
        realtype m_rtol;
        realtype m_get_dx_max_factor;
        bool m_error_outside_bounds;
        realtype m_max_invariant_violation;
        std::vector<realtype> m_special_settings;
        OdeSys(const realtype * const,
               std::vector<realtype>,
               realtype,
               realtype,
               bool, realtype,
               std::vector<realtype>);
        int nrev=0;  // number of calls to roots
        indextype get_ny() const override;
        indextype get_nnz() const override;
        int get_nquads() const override;
        int get_nroots() const override;
        realtype get_dx0(realtype, const realtype * const) override;
        realtype get_dx_max(realtype, const realtype * const) override;
        realtype max_euler_step(realtype, const realtype * const);
        AnyODE::Status rhs(realtype t,
                           const realtype * const ANYODE_RESTRICT y,
                           realtype * const ANYODE_RESTRICT f) override;
        AnyODE::Status dense_jac_cmaj(realtype t,
                                      const realtype * const ANYODE_RESTRICT y,
                                      const realtype * const ANYODE_RESTRICT fy,
                                      realtype * const ANYODE_RESTRICT jac,
                                      long int ldim,
                                      realtype * const ANYODE_RESTRICT dfdt=nullptr) override;
        AnyODE::Status dense_jac_rmaj(realtype t,
                                      const realtype * const ANYODE_RESTRICT y,
                                      const realtype * const ANYODE_RESTRICT fy,
                                      realtype * const ANYODE_RESTRICT jac,
                                      long int ldim,
                                      realtype * const ANYODE_RESTRICT dfdt=nullptr) override;
        AnyODE::Status sparse_jac_csc(realtype t,
                                      const realtype * const ANYODE_RESTRICT y,
                                      const realtype * const ANYODE_RESTRICT fy,
                                      realtype * const ANYODE_RESTRICT data,
                                      indextype * const ANYODE_RESTRICT colptrs,
                                      indextype * const ANYODE_RESTRICT rowvals) override;
        AnyODE::Status jtimes(const realtype * const ANYODE_RESTRICT vec,
                              realtype * const ANYODE_RESTRICT out,
                              realtype t,
                              const realtype * const ANYODE_RESTRICT y,
                              const realtype * const ANYODE_RESTRICT fy) override;

        AnyODE::Status jtimes_setup(realtype /*t*/,
                                    const realtype * const ANYODE_RESTRICT /*y*/,
                                    const realtype * const ANYODE_RESTRICT /*fy*/) override;
        AnyODE::Status roots(realtype t, const realtype * const y, realtype * const out) override;
    };

    OdeSys<realtype, indextype>::OdeSys(const realtype * const params,
                                        std::vector<realtype> atol,
                                        realtype rtol,
                                        realtype get_dx_max_factor,
                                        bool error_outside_bounds,
                                        realtype max_invariant_violation,
                                        std::vector<realtype> special_settings) :
        m_p_cse(${p_common['nsubs']}), m_atol(atol), m_rtol(rtol), m_get_dx_max_factor(get_dx_max_factor),
        m_error_outside_bounds(error_outside_bounds), m_max_invariant_violation(max_invariant_violation),
        m_special_settings(special_settings) {
        m_p.assign(params, params + ${len(p_odesys.params) + p_odesys.ny if p_odesys.append_iv else 0});
        <% idx = 0 %>
      %for cse_token, cse_expr in p_common['cses']:
       %if cse_token.startswith('m_p_cse'):
        ${cse_token} = ${cse_expr}; <% assert cse_token == 'm_p_cse[{0}]'.format(idx); idx += 1 %>
       %else:
        const auto ${cse_token} = ${cse_expr};
       %endif
      %endfor
        use_get_dx_max = (m_get_dx_max_factor > 0.0) ? ${'true' if p_get_dx_max else 'false'} : false;
      %if p_invariants is not None and p_support_recoverable_error:
        if (m_max_invariant_violation != 0.0){
            ${'' if p_odesys.append_iv else 'throw std::runtime_error("append_iv not set to True")'}
            const realtype * const y = params + ${len(p_odesys.params)};
          %for cse_token, cse_expr in p_invariants['cses']:
            const auto ${cse_token} = ${cse_expr};
          %endfor
          %for expr in p_invariants['exprs']:
            m_invar0.push_back(${expr});
          %endfor
       }
      %endif
        ${'\n    '.join(p_constructor)}
    }

    indextype OdeSys<realtype, indextype>::get_ny() const {
        return ${p_odesys.ny};
    }

    indextype OdeSys<realtype, indextype>::get_nnz() const {
        return ${p_odesys.nnz};
    }

    int OdeSys<realtype, indextype>::get_nquads() const {
        return 0;  // Not implemeneted yet (cvodes from Sundials supports this)
    }

    int OdeSys<realtype, indextype>::get_nroots() const {
    %if isinstance(p_nroots, str):
        ${p_nroots}
    %else:
        return ${p_nroots};
    %endif
    }

    AnyODE::Status OdeSys<realtype, indextype>::rhs(realtype x,
                               const realtype * const ANYODE_RESTRICT y,
                               realtype * const ANYODE_RESTRICT f) {
    %if isinstance(p_rhs, str):
        ${p_rhs}
    %else:
        ${'AnyODE::ignore(x);' if p_odesys.autonomous_exprs else ''}
      %for cse_token, cse_expr in p_rhs['cses']:
        const auto ${cse_token} = ${cse_expr};
      %endfor

      %for i, expr in enumerate(p_rhs['exprs']):
        f[${i}] = ${expr};
      %endfor
        this->nfev++;
      %if p_support_recoverable_error:
        if (m_error_outside_bounds){
            if (m_lower_bounds.size() > 0) {
                for (int i=0; i < ${p_odesys.ny}; ++i) {
                    if (y[i] < m_lower_bounds[i]) {
                        std::cerr << "Lower bound (" << m_lower_bounds[0] << ") for "
                                  << (p_odesys_names.size() ? p_odesys_names[i] : std::to_string(i))
                                  << " not fulfilled (" << y[i] << ") at x="<< x << "\n";
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
       %if p_invariants is not None:
        if (m_max_invariant_violation != 0.0){
          %for cse_token, cse_expr in p_invariants['cses']:
            const auto ${cse_token} = ${cse_expr};
          %endfor
          %for idx, invar_expr in enumerate(p_invariants['exprs']):
            if (std::abs(${invar_expr} - m_invar0[${idx}]) > ((m_max_invariant_violation > 0) ? m_max_invariant_violation : std::abs(m_max_invariant_violation*m_invar0[${idx}]) - m_max_invariant_violation)) {
                std::cerr << "Invariant (${idx}) violation at x=" << x << "\n";
                return AnyODE::Status::recoverable_error;
            }
          %endfor
        }
       %endif
      %endif
      %if getattr(p_odesys, '_nonnegative', False) and p_support_recoverable_error:
        for (int i=0; i<${p_odesys.ny}; ++i) if (y[i] < 0) return AnyODE::Status::recoverable_error;
      %endif
        return AnyODE::Status::success;
    %endif
    }


    AnyODE::Status OdeSys<realtype, indextype>::jtimes_setup(
        realtype /*t*/,
        const realtype * const ANYODE_RESTRICT /*y*/,
        const realtype * const ANYODE_RESTRICT /*fy*/) {
        throw std::runtime_error("jtimes_setup not implemented");
    }

    AnyODE::Status OdeSys<realtype, indextype>::jtimes(
                                  const realtype * const ANYODE_RESTRICT v,
                                  realtype * const ANYODE_RESTRICT Jv,
                                  realtype x,
                                  const realtype * const ANYODE_RESTRICT y,
                                  const realtype * const ANYODE_RESTRICT fy) {
    %if p_jtimes is not None:
    %if isinstance(p_jtimes, str):
        ${p_jtimes}
    %else:
        AnyODE::ignore(fy);  // Currently we are not using fy (could be done through extensive pattern matching)
        ${'AnyODE::ignore(x);' if p_odesys.autonomous_exprs else ''}

        %for cse_token, cse_expr in p_jtimes['cses']:
            const auto ${cse_token} = ${cse_expr};
        %endfor

        %for i in range(p_odesys.ny):
            <% curr_expr = p_jtimes['exprs'][i] %>
            Jv[${i}] = ${curr_expr};
        %endfor
    %endif
        this->njvev++;
        return AnyODE::Status::success;
    %else:
        AnyODE::ignore(v); AnyODE::ignore(Jv); AnyODE::ignore(x);
        AnyODE::ignore(y); AnyODE::ignore(fy);
        return AnyODE::Status::unrecoverable_error;
    %endif
    }


    %for order in ('cmaj', 'rmaj'):
    AnyODE::Status OdeSys<realtype, indextype>::dense_jac_${order}(realtype x,
                                          const realtype * const ANYODE_RESTRICT y,
                                          const realtype * const ANYODE_RESTRICT fy,
                                          realtype * const ANYODE_RESTRICT jac,
                                          long int ldim,
                                          realtype * const ANYODE_RESTRICT dfdt) {
    %if p_jac_dense is not None:
    %if order in p_jac_dense:
        ${p_jac_dense[order]}
    %else:
        // The AnyODE::ignore(...) calls below are used to generate code free from false compiler warnings.
        AnyODE::ignore(fy);  // Currently we are not using fy (could be done through extensive pattern matching)
        ${'AnyODE::ignore(x);' if p_odesys.autonomous_exprs else ''}
        ${'AnyODE::ignore(y);' if (not any([yi in p_odesys.get_jac().free_symbols for yi in p_odesys.dep]) and
                                   not any([yi in p_odesys.get_dfdx().free_symbols for yi in p_odesys.dep])) else ''}

      %for cse_token, cse_expr in p_jac_dense['cses']:
        const auto ${cse_token} = ${cse_expr};
      %endfor

      %for i_major in range(p_odesys.ny):
       %for i_minor in range(p_odesys.ny):
    <%
          curr_expr = p_jac_dense['exprs'][i_minor, i_major] if order == 'cmaj' else p_jac_dense['exprs'][i_major, i_minor]
          if curr_expr == '0' and p_jacobian_set_to_zero_by_solver:
              continue
    %>    jac[ldim*${i_major} + ${i_minor}] = ${curr_expr};
       %endfor

      %endfor
        if (dfdt){
          %for idx, expr in enumerate(p_jac_dense['dfdt_exprs']):
            dfdt[${idx}] = ${expr};
          %endfor
        }
        this->njev++;
        return AnyODE::Status::success;
    %endif
    %else:
        // Native code module requires dense_jac_${order} to be defined even if it
        // is never used due to with_jacobian=False
        AnyODE::ignore(x); AnyODE::ignore(y); AnyODE::ignore(fy);
        AnyODE::ignore(jac); AnyODE::ignore(ldim); AnyODE::ignore(dfdt);
        return AnyODE::Status::unrecoverable_error;
    %endif
    }
    %endfor

    realtype OdeSys<realtype, indextype>::get_dx0(realtype x, const realtype * const y) {
    %if p_first_step is None:
        AnyODE::ignore(x); AnyODE::ignore(y);  // avoid compiler warning about unused parameter.
        return 0.0;  // invokes the default behaviour of the chosen solver
    %elif isinstance(p_first_step, str):
        ${p_first_step}
    %else:
      %for cse_token, cse_expr in p_first_step['cses']:
        const realtype ${cse_token} = ${cse_expr};
      %endfor
        ${'' if p_odesys.indep in p_odesys.first_step_expr.free_symbols else 'AnyODE::ignore(x);'}
        ${'' if any([yi in p_odesys.first_step_expr.free_symbols for yi in p_odesys.dep]) else 'AnyODE::ignore(y);'}
        return ${p_first_step['expr']};
    %endif
    }

    AnyODE::Status OdeSys<realtype, indextype>::sparse_jac_csc(realtype x,
                                                               const realtype * const ANYODE_RESTRICT y,
                                                               const realtype * const ANYODE_RESTRICT fy,
                                                               realtype * const ANYODE_RESTRICT data,
                                                               indextype * const ANYODE_RESTRICT colptrs,
                                                               indextype * const ANYODE_RESTRICT rowvals) {
    %if p_jac_sparse is not None:
        AnyODE::ignore(fy);  // Currently we are not using fy (could be done through extensive pattern matching)
        ${'AnyODE::ignore(x);' if p_odesys.autonomous_exprs else ''}
        ${'AnyODE::ignore(y);' if (not any([yi in p_odesys.get_jac().free_symbols for yi in p_odesys.dep]) and
                                   not any([yi in p_odesys.get_dfdx().free_symbols for yi in p_odesys.dep])) else ''}
        %for cse_token, cse_expr in p_jac_sparse['cses']:
            const auto ${cse_token} = ${cse_expr};
        %endfor

        %for i in range(p_odesys.nnz):
          data[${i}] = ${p_jac_sparse['exprs'][i]};
        %endfor

        %for i in range(p_odesys.nnz):
          rowvals[${i}] = ${p_jac_sparse['rowvals'][i]};
        %endfor

        %for i in range(p_odesys.ny + 1):
          colptrs[${i}] = ${p_jac_sparse['colptrs'][i]};
        %endfor
        this->njev++;
        return AnyODE::Status::success;
    %else:
        AnyODE::ignore(x); AnyODE::ignore(y); AnyODE::ignore(fy);
         AnyODE::ignore(data); AnyODE::ignore(colptrs); AnyODE::ignore(rowvals);
        return AnyODE::Status::unrecoverable_error;
    %endif
    }

    realtype OdeSys<realtype, indextype>::get_dx_max(realtype x, const realtype * const y) {
    %if p_get_dx_max is False:
        AnyODE::ignore(x); AnyODE::ignore(y);  // avoid compiler warning about unused parameter.
        return std::numeric_limits<realtype>::max();
    %elif p_get_dx_max is True:
        auto fvec = std::vector<realtype>(${p_odesys.ny});
        auto hvec = std::vector<realtype>(${p_odesys.ny});
        rhs(x, y, &fvec[0]);
        for (indextype idx=0; idx < ${p_odesys.ny}; ++idx){
            if (fvec[idx] == 0) {
                hvec[idx] = std::numeric_limits<realtype>::max();
            } else if (fvec[idx] > 0) {
                hvec[idx] = std::abs(m_upper_bounds[idx] - y[idx])/fvec[idx];
            } else { // fvec[idx] < 0
                hvec[idx] = std::abs((m_lower_bounds[idx] - y[idx])/fvec[idx]);
            }
        }
        const auto result = *std::min_element(std::begin(hvec), std::end(hvec));
        std::cerr << "get_dx_max: " << result << std::endl; //DO-NOT-MERGE!!!
        if (m_get_dx_max_factor == 0.0)
            return result;
        else if (m_get_dx_max_factor < 0.0)
            return -m_get_dx_max_factor*result;
        else
            return m_get_dx_max_factor*result;
    %elif isinstance(p_get_dx_max, str):
        ${p_get_dx_max}
    %else:
        <% raise NotImplementedError("Don't know what to do with: {}".format(p_get_dx_max)) %>
    %endif
    }

    AnyODE::Status OdeSys<realtype, indextype>::roots(realtype x, const realtype * const y, realtype * const out) {
    %if p_roots is None:
        AnyODE::ignore(x); AnyODE::ignore(y); AnyODE::ignore(out);
        return AnyODE::Status::success;
    %elif isinstance(p_roots, str):
        ${p_roots}
    %else:
        ${'' if any(p_odesys.indep in expr.free_symbols for expr in p_odesys.roots) else 'AnyODE::ignore(x);'}

      %for cse_token, cse_expr in p_roots['cses']:
        const auto ${cse_token} = ${cse_expr};
      %endfor

      %for i, expr in enumerate(p_roots['exprs']):
        out[${i}] = ${expr};
      %endfor
        this->nrev++;
        return AnyODE::Status::success;
    %endif
    }
}
