#pragma once
extern "C" void dgemv_(const char* trans, int* m, int* n, const double* alpha, const double* a, int* lda,
                       const double* x, int* incx, const double* beta, double* y, int* incy, int sundials__=0);
extern "C" void sgemv_(const char* trans, int* m, int* n, const float* alpha, const float* a, int* lda,
                       const float* x, int* incx, const float* beta, float* y, int* incy, int sundials__=0);

extern "C" void dgesvd_(const char* jobu, const char* jobvt, int* m, int* n, const double* a,
                        int* lda, double* s, double* u, int* ldu, double* vt, int* ldvt,
                        double* work, int* lwork, int* info );
extern "C" void sgesvd_(const char* jobu, const char* jobvt, int* m, int* n, const float* a,
                        int* lda, float* s, float* u, int* ldu, float* vt, int* ldvt,
                        float* work, int* lwork, int* info );

extern "C" void dgetrf_(const int* dim1, const int* dim2, double* a, int* lda, int* ipiv, int* info);
extern "C" void sgetrf_(const int* dim1, const int* dim2, float* a, int* lda, int* ipiv, int* info);

extern "C" void dgetrs_(const char* trans, const int* n, const int* nrhs, double* a, const int* lda, int* ipiv, double * b, const int* ldb, int* info, int sundials__=0);
extern "C" void sgetrs_(const char* trans, const int* n, const int* nrhs, float* a, const int* lda, int* ipiv, float * b, const int* ldb, int* info, int sundials__=0);

extern "C" void dgbmv_(const char* trans, int* m, int* n, int* kl, int* ku, const double* alpha, const double* a, int* lda,
                       const double* x, int* incx, const double* beta, double* y, int* incy, int sundials__=0);
extern "C" void sgbmv_(const char* trans, int* m, int* n, int* kl, int* ku, const float* alpha, const float* a, int* lda,
                       const float* x, int* incx, const float* beta, float* y, int* incy, int sundials__=0);

extern "C" void dgbtrf_(const int* dim1, const int* dim2, const int* kl, const int* ku, double* a, int* lda, int* ipiv, int* info);
extern "C" void sgbtrf_(const int* dim1, const int* dim2, const int* kl, const int* ku, float* a, int* lda, int* ipiv, int* info);

extern "C" void dgbtrs_(const char* trans, const int* n, const int* kl, const int* ku, const int* nrhs, double* a,
                        const int* lda, int* ipiv, double * b, const int* ldb, int *info, int sundials__=0);
extern "C" void sgbtrs_(const char* trans, const int* n, const int* kl, const int* ku, const int* nrhs, float* a,
                        const int* lda, int* ipiv, float * b, const int* ldb, int*info, int sundials__=0);


#define PROXY_DEFINE(CLS_NAME)                                     \
    template<typename T> struct CLS_NAME ## _callback;

#define PROXY_SPECIALIZATION(CLS_NAME, TYPE, CALLBACK_NAME)        \
    template<> struct CLS_NAME ## _callback<TYPE> {                \
        template<class...Args>                                     \
        void operator()(Args&&... args) const noexcept {           \
            CALLBACK_NAME(std::forward<Args>(args)...);            \
        }                                                          \
    };

namespace AnyODE {

    PROXY_DEFINE(gemv)
    PROXY_SPECIALIZATION(gemv, float, sgemv_)
    PROXY_SPECIALIZATION(gemv, double, dgemv_)

    PROXY_DEFINE(gesvd)
    PROXY_SPECIALIZATION(gesvd, float, sgesvd_)
    PROXY_SPECIALIZATION(gesvd, double, dgesvd_)

    PROXY_DEFINE(getrf)
    PROXY_SPECIALIZATION(getrf, float, sgetrf_)
    PROXY_SPECIALIZATION(getrf, double, dgetrf_)

    PROXY_DEFINE(getrs)
    PROXY_SPECIALIZATION(getrs, float, sgetrs_)
    PROXY_SPECIALIZATION(getrs, double, dgetrs_)

    PROXY_DEFINE(gbmv)
    PROXY_SPECIALIZATION(gbmv, float, sgbmv_)
    PROXY_SPECIALIZATION(gbmv, double, dgbmv_)

    PROXY_DEFINE(gbtrf)
    PROXY_SPECIALIZATION(gbtrf, float, sgbtrf_)
    PROXY_SPECIALIZATION(gbtrf, double, dgbtrf_)

    PROXY_DEFINE(gbtrs)
    PROXY_SPECIALIZATION(gbtrs, float, sgbtrs_)
    PROXY_SPECIALIZATION(gbtrs, double, dgbtrs_)

}

#undef PROXY_DEFINE
#undef PROXY_SPECIALIZATION
