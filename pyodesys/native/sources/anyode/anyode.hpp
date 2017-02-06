#ifdef ANYODE_HPP_D47BAD58870311E6B95F2F58DEFE6E37

#if ANYODE_HPP_D47BAD58870311E6B95F2F58DEFE6E37 != 7
#error "Multiple anyode.hpp files included with version mismatch"
#endif

#else
#define ANYODE_HPP_D47BAD58870311E6B95F2F58DEFE6E37 7


#include <string>
#include <unordered_map>
#include <vector>

namespace AnyODE {
    template<class T> void ignore( const T& ) { } // ignore unused parameter compiler warnings, or: `int /* arg */`

    enum class Status : int {success = 0, recoverable_error = 1, unrecoverable_error = -1};

    struct OdeSysBase {
        int nfev=0, njev=0;
        void * integrator = nullptr;
        std::unordered_map<std::string, int> last_integration_info;
        std::unordered_map<std::string, double> last_integration_info_dbl;
        std::unordered_map<std::string, std::vector<double> > last_integration_info_vecdbl;
        std::unordered_map<std::string, std::vector<int> > last_integration_info_vecint;
        double default_dx0 = 0.0;  // *may* be used by `get_dx0`, 0 signifies solver default
        bool use_get_dx_max = false;  // whether get_dx_max should be called
        bool record_rhs_xvals = false;
        bool record_jac_xvals = false;
        bool record_order = false;
        bool record_fpe = false;
        virtual ~OdeSysBase() {}
        virtual int get_ny() const = 0;
        virtual int get_mlower() const { return -1; } // -1 denotes "not banded"
        virtual int get_mupper() const { return -1; } // -1 denotes "not banded"
        virtual int get_nroots() const { return 0; } // Do not look for roots by default;
        virtual double get_dx0(double /* t */,
                               const double * const /* y */) {
            return default_dx0;
        }
        virtual double get_dx_max(double /* t */, const double * const /* y */) {
            return 0.0;
        }
        virtual Status rhs(double t, const double * const y, double * const f) = 0;
        virtual Status roots(double xval, const double * const y, double * const out) {
            ignore(xval); ignore(y); ignore(out);
            return Status::unrecoverable_error;
        }
        virtual Status dense_jac_cmaj(double t,
                                      const double * const __restrict__ y,
                                      const double * const __restrict__ fy,
                                      double * const __restrict__ jac,
                                      long int ldim,
                                      double * const __restrict__ dfdt=nullptr){
            ignore(t); ignore(y); ignore(fy); ignore(jac); ignore(ldim); ignore(dfdt);
            return Status::unrecoverable_error;
        }
        virtual Status dense_jac_rmaj(double t,
                                      const double * const __restrict__ y,
                                      const double * const __restrict__ fy,
                                      double * const __restrict__ jac,
                                      long int ldim,
                                      double * const __restrict__ dfdt=nullptr){
            ignore(t); ignore(y); ignore(fy); ignore(jac); ignore(ldim); ignore(dfdt);
            return Status::unrecoverable_error;
        }
        virtual Status banded_jac_cmaj(double t,
                                       const double * const __restrict__ y,
                                       const double * const __restrict__ fy,
                                       double * const __restrict__ jac,
                                       long int ldim){
            ignore(t); ignore(y); ignore(fy); ignore(jac); ignore(ldim);
            throw std::runtime_error("banded_jac_cmaj not implemented.");
            return Status::unrecoverable_error;
        }
        virtual Status jac_times_vec(const double * const __restrict__ vec,
                                     double * const __restrict__ out,
                                     double t,
                                     const double * const __restrict__ y,
                                     const double * const __restrict__ fy
                                     )
        {
            ignore(vec);
            ignore(out);
            ignore(t);
            ignore(y);
            ignore(fy);
            return Status::unrecoverable_error;
        }
        virtual Status prec_setup(double t,
                                const double * const __restrict__ y,
                                const double * const __restrict__ fy,
                                bool jok,
                                bool& jac_recomputed,
                                double gamma)
        {
            ignore(t);
            ignore(y);
            ignore(fy);
            ignore(jok);
            ignore(jac_recomputed);
            ignore(gamma);
            return Status::unrecoverable_error;
        }
        virtual Status prec_solve_left(const double t,
                                       const double * const __restrict__ y,
                                       const double * const __restrict__ fy,
                                       const double * const __restrict__ r,
                                       double * const __restrict__ z,
                                       double gamma,
                                       double delta,
                                       const double * const __restrict__ ewt)
        {
            ignore(t);
            ignore(y);
            ignore(fy);
            ignore(r);
            ignore(z);
            ignore(gamma);
            ignore(delta);
            ignore(ewt);
            return Status::unrecoverable_error;
        }
    };

}
#endif /* ANYODE_HPP_D47BAD58870311E6B95F2F58DEFE6E37 */
