#pragma once
#include <string>  // was not properly included in anyode.hpp
#include <unordered_map>  // was not properly included in anyode.hpp
#include <utility>
#include "anyode/anyode.hpp"


struct OdeSys : public AnyODE::OdeSysBase {
    std::vector<double> m_p;
    std::size_t m_nfev, m_njev;
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


namespace {

    // Helper for Cython wrapper
    std::pair<std::pair<std::vector<double>, std::vector<double> >, std::unordered_map<std::string, int> >
    adaptive_return(std::pair<std::vector<double>, std::vector<double> > xout_yout,
                    std::unordered_map<std::string, int> nfo){
        return std::make_pair(xout_yout, nfo);
    }
}
