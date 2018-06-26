#ifdef ANYODE_HPP_D47BAD58870311E6B95F2F58DEFE6E37

#if ANYODE_HPP_D47BAD58870311E6B95F2F58DEFE6E37 != 16
#error "Multiple anyode.hpp files included with version mismatch"
#endif

#else
#define ANYODE_HPP_D47BAD58870311E6B95F2F58DEFE6E37 16

#ifndef ANYODE_RESTRICT
  #if defined(__GNUC__)
    #define ANYODE_RESTRICT __restrict__
  #elif defined(_MSC_VER) && _MSC_VER >= 1400
    #define ANYODE_RESTRICT __restrict
  #elif defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
    #define ANYODE_RESTRICT restrict
  #else
    #define ANYODE_RESTRICT
  #endif
#endif


#include <memory>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>
#include <anyode/anyode_util.hpp>

BEGIN_NAMESPACE(AnyODE)
struct Info {
    std::unordered_map<std::string, int> nfo_int = {};
    std::unordered_map<std::string, double> nfo_dbl = {};
    std::unordered_map<std::string, std::vector<double> > nfo_vecdbl = {};
    std::unordered_map<std::string, std::vector<int> > nfo_vecint = {};
    void clear() {
        nfo_int.clear();
        nfo_dbl.clear();
        nfo_vecdbl.clear();
        nfo_vecint.clear();
    }
    void update(
        const std::unordered_map<std::string, int> &new_int,
        const std::unordered_map<std::string, double> &new_dbl,
        const std::unordered_map<std::string, std::vector<double> > &new_vecdbl,
        const std::unordered_map<std::string, std::vector<int> > &new_vecint)
    {
#define ANYODE_INCREMENT(TOKEN)                     \
        for (const auto &kv : new_ ## TOKEN){       \
            const auto &k = kv.first;               \
            const auto &v = kv.second;              \
            nfo_ ## TOKEN[k] += v; \
        }
        ANYODE_INCREMENT(int);
        ANYODE_INCREMENT(dbl);
#undef ANYODE_INCREMENT
#define ANYODE_APPEND(TOKEN)                                            \
        for (const auto &kv : new_vec ## TOKEN){                        \
            const auto &source = kv.second;                             \
            auto &dest = nfo_vec ## TOKEN[kv.first];                    \
            extend_vec(dest, source);                                   \
        }
        ANYODE_APPEND(int);
        ANYODE_APPEND(dbl);
#undef ANYODE_APPEND
    }
    template<typename stream_t>
    void dump_ascii(stream_t& out, const std::string &joiner, const std::string &delimiter) const {
#define ANYODE_PRINT(DICT_OF_SCALARS)               \
        for (const auto &kv : DICT_OF_SCALARS){     \
            const auto &k = kv.first;               \
            const auto &v = kv.second;              \
            out << k << joiner << v << delimiter;   \
        }
        ANYODE_PRINT(nfo_int);
        ANYODE_PRINT(nfo_dbl);
#undef ANYODE_PRINT
#define ANYODE_PRINT(DICT_OF_VECTORS)               \
        for (const auto &kv : DICT_OF_VECTORS){     \
            const auto &k = kv.first;               \
            const auto &v = kv.second;              \
            out << k << joiner << "[";              \
            for (auto it=v.begin();it != v.end();){ \
                out << *it;                         \
                ++it;                               \
                if (it == v.end()){                 \
                    break;                          \
                } else {                            \
                    out << delimiter;               \
                }                                   \
            }                                       \
            out << "]" << delimiter;                \
        }
        ANYODE_PRINT(nfo_vecdbl);
        ANYODE_PRINT(nfo_vecint);
#undef ANYODE_PRINT
    }
};


struct Result {
    int nt, ny, nquads, nroots;
    Info info;
private:
    std::unique_ptr<double[], decltype(std::free) *> m_data;
public:
    Result() = delete;
    Result(const Result&) = delete;
    Result(int nt, int ny, int nquads, int nroots, double * data) :
        nt(nt), ny(ny), nquads(nquads), nroots(nroots), m_data(data, std::free)
    {
    }
    double &t (int tidx) { return m_data[tidx*(nquads+ny+1)]; }
    double &y (int tidx, int yidx) { return m_data[tidx*(nquads+ny+1) + 1 + yidx]; }
    double &q (int tidx, int qidx) { return m_data[tidx*(nquads+ny+1) + 1 + ny + qidx]; }
    double * get_raw_ptr() const { return m_data.get(); }
    template<typename stream_t>
    void dump_ascii(stream_t& out) {
        const auto nyq = ny + nquads;
        for (int ti=0; ti<nt; ++ti){
            out << t(ti);
            for (int yqi=0; yqi < nyq; ++yqi){
                out << " " << y(ti, yqi);
            }
            out << '\n';
        }
    }
};

template<class T> void ignore( const T& ) { } // ignore unused parameter compiler warnings, or: `int /* arg */`

enum class Status : int {success = 0, recoverable_error = 1, unrecoverable_error = -1};

template <typename Real_t=double>
struct OdeSysBase {
    int nfev=0, njev=0;
    void * integrator = nullptr;
    void * user_data = nullptr;  // for those who don't want to subclass
    Info current_info;
    Real_t default_dx0 = 0.0;  // *may* be used by `get_dx0`, 0 signifies solver default
    bool autonomous_exprs = false;
    bool use_get_dx_max = false;  // whether get_dx_max should be called
    bool record_rhs_xvals = false;
    bool record_jac_xvals = false;
    bool record_order = false;
    bool record_fpe = false;
    bool record_steps = false;
    virtual ~OdeSysBase() {}
    virtual int get_ny() const = 0;
    virtual int get_mlower() const { return -1; } // -1 denotes "not banded"
    virtual int get_mupper() const { return -1; } // -1 denotes "not banded"
    virtual int get_nquads() const { return 0; } // Do not track quadratures by default;
    virtual int get_nroots() const { return 0; } // Do not look for roots by default;
    virtual Real_t get_dx0(Real_t /* t */,
                           const Real_t * const /* y */) {
        return default_dx0;
    }
    virtual Real_t get_dx_max(Real_t /* t */, const Real_t * const /* y */) {
        return 0.0;
    }
    virtual Status rhs(Real_t t, const Real_t * const y, Real_t * const f) = 0;
    virtual Status quads(Real_t xval, const Real_t * const y, Real_t * const out) {
        ignore(xval); ignore(y); ignore(out);
        return Status::unrecoverable_error;
    }
    virtual Status roots(Real_t xval, const Real_t * const y, Real_t * const out) {
        ignore(xval); ignore(y); ignore(out);
        return Status::unrecoverable_error;
    }
    virtual Status dense_jac_cmaj(Real_t t,
                                  const Real_t * const ANYODE_RESTRICT y,
                                  const Real_t * const ANYODE_RESTRICT fy,
                                  Real_t * const ANYODE_RESTRICT jac,
                                  long int ldim,
                                  Real_t * const ANYODE_RESTRICT dfdt=nullptr){
        ignore(t); ignore(y); ignore(fy); ignore(jac); ignore(ldim); ignore(dfdt);
        return Status::unrecoverable_error;
    }
    virtual Status dense_jac_rmaj(Real_t t,
                                  const Real_t * const ANYODE_RESTRICT y,
                                  const Real_t * const ANYODE_RESTRICT fy,
                                  Real_t * const ANYODE_RESTRICT jac,
                                  long int ldim,
                                  Real_t * const ANYODE_RESTRICT dfdt=nullptr){
        ignore(t); ignore(y); ignore(fy); ignore(jac); ignore(ldim); ignore(dfdt);
        return Status::unrecoverable_error;
    }
    virtual Status banded_jac_cmaj(Real_t t,
                                   const Real_t * const ANYODE_RESTRICT y,
                                   const Real_t * const ANYODE_RESTRICT fy,
                                   Real_t * const ANYODE_RESTRICT jac,
                                   long int ldim){
        ignore(t); ignore(y); ignore(fy); ignore(jac); ignore(ldim);
        throw std::runtime_error("banded_jac_cmaj not implemented.");
        return Status::unrecoverable_error;
    }
    virtual Status jac_times_vec(const Real_t * const ANYODE_RESTRICT vec,
                                 Real_t * const ANYODE_RESTRICT out,
                                 Real_t t,
                                 const Real_t * const ANYODE_RESTRICT y,
                                 const Real_t * const ANYODE_RESTRICT fy
                                 )
    {
        ignore(vec);
        ignore(out);
        ignore(t);
        ignore(y);
        ignore(fy);
        return Status::unrecoverable_error;
    }
    virtual Status prec_setup(Real_t t,
                            const Real_t * const ANYODE_RESTRICT y,
                            const Real_t * const ANYODE_RESTRICT fy,
                            bool jok,
                            bool& jac_recomputed,
                            Real_t gamma)
    {
        ignore(t);
        ignore(y);
        ignore(fy);
        ignore(jok);
        ignore(jac_recomputed);
        ignore(gamma);
        return Status::unrecoverable_error;
    }
    virtual Status prec_solve_left(const Real_t t,
                                   const Real_t * const ANYODE_RESTRICT y,
                                   const Real_t * const ANYODE_RESTRICT fy,
                                   const Real_t * const ANYODE_RESTRICT r,
                                   Real_t * const ANYODE_RESTRICT z,
                                   Real_t gamma,
                                   Real_t delta,
                                   const Real_t * const ANYODE_RESTRICT ewt)
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
END_NAMESPACE(AnyODE)

#endif /* ANYODE_HPP_D47BAD58870311E6B95F2F58DEFE6E37 */
