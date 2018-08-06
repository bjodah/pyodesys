#pragma once
#include "anyode/anyode.hpp"

namespace odesys_anyode {
template<typename Real_t=double, typename Index_t=int>
    struct OdeSys : public AnyODE::OdeSysBase<Real_t, Index_t> {
        std::vector<Real_t> m_p;
        std::vector<Real_t> m_p_cse;
        std::vector<Real_t> m_atol;
        std::vector<Real_t> m_upper_bounds;
        std::vector<Real_t> m_lower_bounds;
        std::vector<Real_t> m_invar0;
        double m_rtol;
        Real_t m_get_dx_max_factor;
        bool m_error_outside_bounds;
        Real_t m_max_invariant_violation;
        std::vector<Real_t> m_special_settings; // for user specialization of code
        OdeSys(const Real_t * const, std::vector<Real_t>, Real_t, Real_t,
               bool, Real_t, std::vector<Real_t>);
        int nrev=0;  // number of calls to roots
        Index_t get_ny() const override;
        int get_nquads() const override;
        int get_nroots() const override;
        Real_t get_dx0(Real_t, const Real_t * const) override;
        Real_t get_dx_max(Real_t, const Real_t * const) override;
        AnyODE::Status rhs(Real_t x,
                           const Real_t * const __restrict__ y,
                           Real_t * const __restrict__ f) override;
        AnyODE::Status jtimes(const Real_t * const __restrict__ v,
                              Real_t * const __restrict__ Jv,
                              Real_t x,
                              const Real_t * const __restrict__ y,
                              const Real_t * const __restrict__ fy) override;
        AnyODE::Status dense_jac_cmaj(Real_t x,
                                      const Real_t * const __restrict__ y,
                                      const Real_t * const __restrict__ fy,
                                      Real_t * const __restrict__ jac,
                                      long int ldim,
                                      Real_t * const __restrict__ dfdt=nullptr) override;
        AnyODE::Status dense_jac_rmaj(Real_t x,
                                      const Real_t * const __restrict__ y,
                                      const Real_t * const __restrict__ fy,
                                      Real_t * const __restrict__ jac,
                                      long int ldim,
                                      Real_t * const __restrict__ dfdt=nullptr) override;
        AnyODE::Status roots(Real_t x, const Real_t * const y, Real_t * const out) override;
    };
}
