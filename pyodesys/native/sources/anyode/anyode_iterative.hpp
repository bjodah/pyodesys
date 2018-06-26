#pragma once

#include <memory>

#include <anyode/anyode.hpp>
#include <anyode/anyode_buffer.hpp> // make_unique
#include <anyode/anyode_blas_lapack.hpp>  // dgemv, dgesvd
#include <anyode/anyode_matrix.hpp> // DenseMatrix
#include <anyode/anyode_decomposition.hpp>  // SVD

namespace AnyODE {

    template <typename Real_t=double, typename JacMat_t=DenseMatrix<Real_t>, typename Decomp_t=SVD<Real_t>>
    struct OdeSysIterativeBase : public OdeSysBase<Real_t> {
        int m_njacvec_dot=0, m_nprec_setup=0, m_nprec_solve=0;
        std::unique_ptr<JacMat_t> m_jac_cache {nullptr};
        std::unique_ptr<JacMat_t> m_M_cache {nullptr};
        std::unique_ptr<Decomp_t> m_decomp_cache {nullptr};

        virtual Status jac_times_vec(const Real_t * const ANYODE_RESTRICT vec,
                                     Real_t * const ANYODE_RESTRICT out,
                                     Real_t t,
                                     const Real_t * const ANYODE_RESTRICT y,
                                     const Real_t * const ANYODE_RESTRICT fy
                                     ) override
        {
            // See "Jacobian information (matrix-vector product)"
            //     (4.6.8 in cvs_guide.pdf for sundials 2.7.0)
            auto status = AnyODE::Status::success;
            const int ny = this->get_ny();
            auto jac = make_unique<JacMat_t>(nullptr, ny, ny, ny);
            jac->set_to(0.0);
            status = this->dense_jac_cmaj(t, y, fy, jac->m_data, jac->m_ld);
            jac->dot_vec(vec, out);
            m_njacvec_dot++;
            return status;
        }

        virtual Status prec_setup(Real_t t,
                                  const Real_t * const ANYODE_RESTRICT y,
                                  const Real_t * const ANYODE_RESTRICT fy,
                                  bool jac_ok,
                                  bool& jac_recomputed,
                                  Real_t gamma) override
        {
            const int ny = this->get_ny();
            auto status = AnyODE::Status::success;
            ignore(gamma);
            // See "Preconditioning (Jacobian data)" in cvs_guide.pdf (4.6.10 for 2.7.0)
            if (m_jac_cache == nullptr)
                m_jac_cache = make_unique<JacMat_t>(nullptr, ny, ny, ny);

            if (jac_ok){
                jac_recomputed = false;
            } else {
                status = this->dense_jac_cmaj(t, y, fy, m_jac_cache->m_data, m_jac_cache->m_ld);
                jac_recomputed = true;
            }
            if (m_M_cache == nullptr)
                m_M_cache = make_unique<JacMat_t>(nullptr, ny, ny, ny);
            m_M_cache->set_to_eye_plus_scaled_mtx(-gamma, *m_jac_cache);
            m_decomp_cache = make_unique<Decomp_t>(m_M_cache.get());
            m_decomp_cache->factorize();
            m_nprec_setup++;
            return status;
        }

        virtual Status prec_solve_left(const Real_t t,
                                       const Real_t * const ANYODE_RESTRICT y,
                                       const Real_t * const ANYODE_RESTRICT fy,
                                       const Real_t * const ANYODE_RESTRICT r,
                                       Real_t * const ANYODE_RESTRICT z,
                                       Real_t /* gamma */,
                                       Real_t /* delta */,
                                       const Real_t * const ANYODE_RESTRICT ewt
                                       ) override
        {
            // See 4.6.9 on page 75 in cvs_guide.pdf (Sundials 2.6.2)
            // Solves P*z = r, where P ~= I - gamma*J
            if (ewt)
                throw std::runtime_error("Not implemented: ewt in prec_solve_left");
            m_nprec_solve++;

            ignore(t); ignore(fy); ignore(y);
            int info = m_decomp_cache->solve(r, z);
            if (info == 0)
                return AnyODE::Status::success;
            return AnyODE::Status::recoverable_error;
        }


    };
}
