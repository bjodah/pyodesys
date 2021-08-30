#pragma once
#include "anyode/anyode_iterative.hpp"

namespace odesys_anyode {
    template<typename Real_t=double, typename Index_t=int>
    struct OdeSys : public AnyODE::OdeSysIterativeBase<Real_t, Index_t> {
        std::vector<Real_t> m_p;
        std::vector<Real_t> m_cse;
        std::vector<Real_t> m_atol;
        std::vector<Real_t> m_upper_bounds;
        std::vector<Real_t> m_lower_bounds;
        std::vector<Real_t> m_invar;
        std::vector<Real_t> m_invar0;
        Real_t m_rtol;
        Real_t m_get_dx_max_factor;
        bool m_error_outside_bounds;
        Real_t m_max_invariant_violation;
        std::vector<Real_t> m_special_settings;
        OdeSys(const Real_t * const, std::vector<Real_t>, Real_t, Real_t,
               bool, Real_t, std::vector<Real_t>);
        int nrev=0;  // number of calls to roots
        Index_t get_ny() const override;
        Index_t get_nnz() const override;
        int get_nquads() const override;
        int get_nroots() const override;
        Real_t get_dx0(Real_t, const Real_t * const) override;
        Real_t get_dx_max(Real_t, const Real_t * const) override;
        Real_t max_euler_step(Real_t, const Real_t * const);
        AnyODE::Status rhs(Real_t t,
                           const Real_t * const y,
                           Real_t * const f) override;
        AnyODE::Status dense_jac_cmaj(Real_t t,
                                      const Real_t * const y,
                                      const Real_t * const fy,
                                      Real_t * const jac,
                                      long int ldim,
                                      Real_t * const dfdt=nullptr) override;
        AnyODE::Status dense_jac_rmaj(Real_t t,
                                      const Real_t * const y,
                                      const Real_t * const fy,
                                      Real_t * const jac,
                                      long int ldim,
                                      Real_t * const dfdt=nullptr) override;
        AnyODE::Status sparse_jac_csc(Real_t t,
                                      const Real_t * const y,
                                      const Real_t * const fy,
                                      Real_t * const data,
                                      Index_t * const colptrs,
                                      Index_t * const rowvals) override;
        AnyODE::Status jtimes(const Real_t * const vec,
                              Real_t * const out,
                              Real_t t,
                              const Real_t * const y,
                              const Real_t * const fy) override;
        AnyODE::Status jtimes_setup(Real_t t,
                          const Real_t * const y,
                          const Real_t * const fy) override;
        AnyODE::Status roots(Real_t t, const Real_t * const y, Real_t * const out) override;
    };
}
