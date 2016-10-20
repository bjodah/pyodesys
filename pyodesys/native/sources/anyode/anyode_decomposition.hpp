#ifndef ANYODE_DECOMPOSITION_HPP_3BD9D4BE95FC11E6A79D1F63EAC57037
#define ANYODE_DECOMPOSITION_HPP_3BD9D4BE95FC11E6A79D1F63EAC57037

#include <cmath>

#include "anyode/anyode_matrix.hpp"
#include "anyode/anyode_buffer.hpp"

namespace AnyODE {
    struct DecompositionBase {
        virtual int factorize() = 0;
        virtual int solve(const double* const, double * const) = 0;
    };

    struct SVD : public DecompositionBase {
        DenseMatrixView * m_view;
        buffer_t<double> m_s;
        int m_ldu;
        buffer_t<double> m_u;
        int m_ldvt;
        buffer_t<double> m_vt;
        buffer_t<double> m_work;
        int m_lwork = -1; // Query
        int m_info;
        double m_condition_number = -1;

        SVD(DenseMatrixView * view) :
            m_view(view), m_s(buffer_factory<double>(std::min(view->m_nr, view->m_nc))),
            m_ldu(view->m_nr), m_u(buffer_factory<double>(m_ldu*(view->m_nr))),
            m_ldvt(view->m_nc), m_vt(buffer_factory<double>(m_ldvt*(view->m_nc)))
        {
            int info;
            double optim_work_size;
            char mode = 'A';
            dgesvd_(&mode, &mode, &(m_view->m_nr), &(m_view->m_nc), m_view->m_data, &(m_view->m_ld),
                    buffer_get_raw_ptr(m_s), buffer_get_raw_ptr(m_u), &m_ldu,
                    buffer_get_raw_ptr(m_vt), &m_ldvt, &optim_work_size, &m_lwork, &info);
            m_lwork = static_cast<int>(optim_work_size);
            m_work = buffer_factory<double>(m_lwork);
            m_info = factorize();
        }
        int factorize() override{
            int info;
            char mode = 'A';
            dgesvd_(&mode, &mode, &(m_view->m_nr), &(m_view->m_nc), m_view->m_data, &(m_view->m_ld),
                    buffer_get_raw_ptr(m_s), buffer_get_raw_ptr(m_u), &m_ldu,
                    buffer_get_raw_ptr(m_vt), &m_ldvt, buffer_get_raw_ptr(m_work), &m_lwork, &info);

            m_condition_number = std::fabs(m_s[0]/m_s[std::min(m_view->m_nr, m_view->m_nc) - 1]);
            return info;
        }
        int solve(const double* const b, double * const x) override{
            double alpha=1, beta=0;
            int incx=1, incy=1;
            char trans = 'T';
            int sundials_dummy = 0;
            auto y1 = buffer_factory<double>(m_view->m_nr);
            dgemv_(&trans, &(m_view->m_nr), &(m_view->m_nr), &alpha, buffer_get_raw_ptr(m_u), &(m_ldu),
                   const_cast<double*>(b), &incx, &beta, buffer_get_raw_ptr(y1), &incy,
                   sundials_dummy);
            for (int i=0; i < m_view->m_nr; ++i)
                y1[i] /= m_s[i];
            dgemv_(&trans, &(m_view->m_nc), &(m_view->m_nc), &alpha, buffer_get_raw_ptr(m_vt), &m_ldvt,
                   buffer_get_raw_ptr(y1), &incx, &beta, x, &incy,
                   sundials_dummy);
            return 0;
        }

    };

}

#endif /* ANYODE_DECOMPOSITION_HPP_3BD9D4BE95FC11E6A79D1F63EAC57037 */
