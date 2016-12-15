#pragma once
#include "anyode/anyode_iterative.hpp"

namespace odesys_anyode {
    struct OdeSys : public AnyODE::OdeSysIterativeBase {
        std::vector<double> m_p;
        std::vector<double> m_atol;
        double m_rtol;
        int nrev=0;  // number of calls to roots
        OdeSys(const double * const, std::vector<double>, double);
        int get_ny() const override;
        int get_nroots() const override;
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
        double get_dx0(double, const double * const) override;
        AnyODE::Status roots(double t, const double * const y, double * const out) override;
    };
}
