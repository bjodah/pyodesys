#ifndef ANYODE_ITERATIVE_HPP_42E48E8295F411E6BFBC5726D640C316
#define ANYODE_ITERATIVE_HPP_42E48E8295F411E6BFBC5726D640C316

#include <memory>

#include <anyode/anyode.hpp>
#include <anyode/anyode_blas_lapack.hpp>  // dgemv, dgesvd
#include <anyode/anyode_matrix.hpp> // DenseMatrixView
#include <anyode/anyode_buffer.hpp>  // make_unique
#include <anyode/anyode_decomposition.hpp>  // SVD

namespace AnyODE {

    struct OdeSysIterativeBase : public OdeSysBase {
        int njacvec_dot=0, nprec_setup=0, nprec_solve=0;
        std::unique_ptr<MatrixView> m_jac_cache {nullptr};
        std::unique_ptr<MatrixView> m_prec_cache {nullptr};
        bool m_update_prec_cache = false;
        double m_old_gamma;

        virtual Status jac_times_vec(const double * const __restrict__ vec,
                                     double * const __restrict__ out,
                                     double t,
                                     const double * const __restrict__ y,
                                     const double * const __restrict__ fy
                                     ) override
        {
            // See 4.6.7 on page 67 (77) in cvs_guide.pdf (Sundials 2.5)
            auto status = AnyODE::Status::success;
            const int ny = this->get_ny();
            if (m_jac_cache == nullptr){
                m_jac_cache = make_unique<DenseMatrixView>(nullptr, ny, ny, ny, true);
                status = this->dense_jac_cmaj(t, y, fy, m_jac_cache->m_data, m_jac_cache->m_ld);
            }
            m_jac_cache->dot_vec(vec, out);
            njacvec_dot++;
            return status;
        }

        virtual Status prec_setup(double t,
                                const double * const __restrict__ y,
                                const double * const __restrict__ fy,
                                bool jok,
                                bool& jac_recomputed,
                                double gamma) override
        {
            const int ny = this->get_ny();
            auto status = AnyODE::Status::success;
            ignore(gamma);
            // See 4.6.9 on page 68 (78) in cvs_guide.pdf (Sundials 2.5)
            if (m_jac_cache == nullptr)
                m_jac_cache = make_unique<DenseMatrixView>(nullptr, ny, ny, ny, true);
            if (!jok){
                status = this->dense_jac_cmaj(t, y, fy, m_jac_cache->m_data, m_jac_cache->m_ld);
                m_update_prec_cache = true;
                jac_recomputed = true;
            } else {
                jac_recomputed = false;
            }
            nprec_setup++;
            return status;
        }

        virtual Status prec_solve_left(const double t,
                                       const double * const __restrict__ y,
                                       const double * const __restrict__ fy,
                                       const double * const __restrict__ r,
                                       double * const __restrict__ z,
                                       double gamma,
                                       double delta,
                                       const double * const __restrict__ ewt
                                       ) override
        {
            // See 4.6.9 on page 75 in cvs_guide.pdf (Sundials 2.6.2)
            // Solves P*z = r, where P ~= I - gamma*J
            ignore(delta);
            const int ny = this->get_ny();
            if (ewt)
                throw std::runtime_error("Not implemented.");
            nprec_solve++;

            ignore(t); ignore(fy); ignore(y);
            bool recompute = false;
            if (m_prec_cache == nullptr){
                m_prec_cache = make_unique<DenseMatrixView>(nullptr, ny, ny, ny, true);
                recompute = true;
            } else {
                if (m_update_prec_cache or (m_old_gamma != gamma))
                    recompute = true;
            }
            if (recompute){
                m_old_gamma = gamma;
                m_prec_cache->set_to_eye_plus_scaled_mtx(-gamma, *m_jac_cache);
            }
            int info;
            auto decomp = SVD((DenseMatrixView*)(m_prec_cache.get()));
            info = decomp.solve(r, z);
            if (info == 0)
                return AnyODE::Status::success;
            return AnyODE::Status::recoverable_error;
        }


    };
}

#endif /* ANYODE_ITERATIVE_HPP_42E48E8295F411E6BFBC5726D640C316 */
