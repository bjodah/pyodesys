#pragma once
#include "anyode/anyode.hpp"

namespace odesys_anyode {
    struct OdeSys : public AnyODE::OdeSysBase {
        std::vector<double> m_p;
        OdeSys(const double * const params);
        int get_ny() const override;
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
    };
}
