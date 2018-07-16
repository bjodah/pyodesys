#pragma once
#include <cmath>
#include <functional>
using namespace std;

namespace AnyODE {

    template <typename Real_t = double> struct getrf_callback {
        void operator()(const int * nr, const int * nc, Real_t * a,
                        int * lda, int * ipiv, int * info) const noexcept {
            // Unblocked algorithm for LU decomposition general matrices
            // employing Doolittle's algorithm with rowswaps.
            //
            // ipiv indexing starts at 1 (Fortran compatibility)
            *info = 0;
            const int dim = std::min(*nr, *nc);
            if (dim == 0) return;

            auto A = [&](int ri, int ci) -> Real_t& { return a[ci*(*lda) + ri]; };
            auto swaprows = [&](int ri1, int ri2) { // this is not cache friendly
                for (int ci=0; ci<dim; ++ci)
                    std::swap(A(ri1, ci), A(ri2, ci));
            };

            for (int i=0; i < dim; ++i) {
                int pivrow = i;
                Real_t absmax = std::abs(A(i, i));
                for (int j=i+1; j<*nr; ++j) {
                    // Find pivot
                    Real_t curabs = std::abs(A(j, i));
                    if (curabs > absmax){
                        absmax = curabs;
                        pivrow = j;
                    }
                }
                if ((absmax == 0) && (*info == 0))
                    *info = pivrow+1;
                ipiv[i] = pivrow+1;
                if (pivrow != i) {
                    // Swap rows
                    swaprows(i, pivrow);
                }
                // Eliminate in column
                for (int ri=i+1; ri<*nr; ++ri){
                    A(ri, i) /= A(i, i);
                }
                // Subtract from rows
                for (int ci=i+1; ci<*nc; ++ci){
                    const Real_t A_i_ci = A(i, ci);
                    for (int ri=i+1; ri<*nr; ++ri){
                        A(ri, ci) -= A(ri, i)*A_i_ci;
                    }
                }
            }
            ipiv[dim-1] = dim;
        }
    };


    template <typename Real_t = double> struct getrs_callback {
        void operator()(const char * trans, const int * n, const int * nrhs, Real_t * a,
                        const int * lda, int * ipiv, Real_t * b, const int * ldb, int * info,
                        int sundials__=0) const noexcept {

            ignore(trans);
            ignore(sundials__);
            *info = 0;
            if (*n < 0)  *info = -1;
            if (*nrhs < 0)  *info = -2;
            if (a == nullptr)  *info = -3;
            if (*lda < 0)  *info = -4;
            if (ipiv == nullptr)  *info = -5;
            if (b == nullptr) *info = -6;
            if (*ldb < 0) *info = -7;
            if (*info != 0 || n == 0)
                return;
            auto A = [&](const int ri, const int ci) -> Real_t& { return a[ri + ci*(*lda)]; };
            auto B = [&](const int ri, const int idx) -> Real_t& { return b[ri + idx*(*ldb)]; };
            for (int k=0; k<*nrhs; ++k){
                for (int i=0; i<*n; ++i)
                    if (ipiv[i]-1 != i)
                        std::swap(B(i, k), B(ipiv[i]-1, k));
                for (int i=1; i<*n; ++i){
                    for (int j=0; j<i; ++j)
                        B(i, k) -= A(i, j)*B(j, k);
                }
                for (int i=*n-1; i>=0; --i){
                    for (int j=i+1; j<*n; ++j)
                        B(i, k) -= A(i, j)*B(j, k);
                    B(i, k) /= A(i, i);
                }
            }
        }
    };

    template <typename Real_t = double> struct gemv_callback {
        void operator()(const char* trans, int * m, int * n, const Real_t * alpha,
                        Real_t * a, int* lda, const Real_t * x, int * incx,
                        const Real_t * beta, Real_t * y, int * incy, int sundials__=0) const noexcept {
            ignore(incx);
            ignore(incy);
            ignore(sundials__);
            std::function<Real_t& (const int, const int)> A;
            if (*trans == 'T')
                A = [&](const int ri, const int ci) -> Real_t& { return a[ri*(*lda) + ci]; };
            else
                A = [&](const int ri, const int ci) -> Real_t& { return a[ci*(*lda) + ri]; };

            int i, j;
            Real_t y0;

            for (i=0; i != *m; i++) {
                y0 = (*beta) * y[i];
                for (j=0; j != *n; j++) y0 += *alpha * A(i, j) * x[j];
                y[i] = y0;
            }
        }
    };
}