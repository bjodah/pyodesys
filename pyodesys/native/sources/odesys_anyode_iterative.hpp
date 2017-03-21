#pragma once
#include "anyode/anyode_iterative.hpp"

namespace odesys_anyode {
    struct OdeSys : public AnyODE::OdeSysIterativeBase {
        std::vector<double> m_p;
        std::vector<double> m_p_cse;
        std::vector<double> m_atol;
        std::vector<double> m_upper_bounds;
        std::vector<double> m_lower_bounds;
        std::vector<double> m_invar0;
        double m_rtol;
        double m_get_dx_max_factor;
        bool m_error_outside_bounds;
        double m_max_invariant_violation;
        std::vector<double> m_special_settings;
        OdeSys(const double * const, std::vector<double>, double, double,
               bool, double, std::vector<double>);
        int nrev=0;  // number of calls to roots
        int get_ny() const override;
        int get_nroots() const override;
        double get_dx0(double, const double * const) override;
        double get_dx_max(double, const double * const) override;
        double max_euler_step(double, const double * const);
        AnyODE::Status rhs(double t,
                           const double * const __restrict__ y,
                           double * const __restrict__ f) override;
        AnyODE::Status dense_jac_cmaj(double t,
                                      const double * const __restrict__ y,
                                      const double * const __restrict__ fy,
                                      double * const __restrict__ jac,
                                      long int ldim,
                                      double * const __restrict__ dfdt=nullptr) override;
        AnyODE::Status dense_jac_rmaj(double t,
                                      const double * const __restrict__ y,
                                      const double * const __restrict__ fy,
                                      double * const __restrict__ jac,
                                      long int ldim,
                                      double * const __restrict__ dfdt=nullptr) override;
        AnyODE::Status roots(double t, const double * const y, double * const out) override;
    };
}
