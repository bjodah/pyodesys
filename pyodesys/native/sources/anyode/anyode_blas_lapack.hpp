#ifndef ANYODE_BLAS_LAPACK_HPP3FDF14C295FA11E6A327373BE69EEE93
#define ANYODE_BLAS_LAPACK_HPP3FDF14C295FA11E6A327373BE69EEE93

extern "C" void dgemv_(const char* trans, int* m, int* n, const double* alpha, const double* a, int* lda,
                       const double* x, int* incx, const double* beta, double* y, int* incy, int stupid_sundials=0);

extern "C" void dgesvd_(const char* jobu, const char* jobvt, int* m, int* n, const double* a,
                       int* lda, double* s, double* u, int* ldu, double* vt, int* ldvt,
                       double* work, int* lwork, int* info );

#endif /* ANYODE_BLAS_LAPACK_HPP3FDF14C295FA11E6A327373BE69EEE93 */
