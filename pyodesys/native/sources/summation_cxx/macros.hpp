#pragma once
#ifndef SXX_RESTRICT
#if defined(__GNUC__)
#define SXX_RESTRICT __restrict__
#elif defined(_MSC_VER) && _MSC_VER >= 1400
#define SXX_RESTRICT __restrict
// #elif defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
//   #define SXX_RESTRICT restrict
#else
#define SXX_RESTRICT
#endif
#endif

#ifndef SXX_PREFER_INLINE
#if defined(__GNUC__)
#define SXX_PREFER_INLINE __attribute__((flatten))
#elif defined(_MSC_VER) && _MSC_VER >= 1400
#define SXX_PREFER_INLINE __forceinline
#else
#define SXX_PREFER_INLINE
#endif
#endif

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
#include <concepts>
#define SXX_FWD_IT_CONCEPT std::forward_iterator
#define SXX_RND_IT_CONCEPT std::random_access_iterator
#define SXX_UNSIGNED_INTEGRAL std::unsigned_integral
#else
#define SXX_FWD_IT_CONCEPT typename
#define SXX_RND_IT_CONCEPT typename
#define SXX_UNSIGNED_INTEGRAL typename
#endif

// Math macros to support e.g. __float128 without std lib support:
#ifndef SXX_ABS
#define SXX_ABS(x) (((x) < 0) ? -(x) : (x))
#endif

#ifndef SXX_NEUMAIER_BRANCH
// see test/bench.cpp
#define SXX_NEUMAIER_BRANCH 1
#endif

#if defined(SXX_NEUMAIER_SWAP)
#error "API has changed, update your compilation flags accordingly"
#endif

#define SXX_SWP_TMP_(ACCUM, CARRY, ELEM, TYPE, TMP, DO_SWAP)                   \
    TYPE TMP;                                                                       \
    if (DO_SWAP) {                                                                  \
        if ((CARRY) == 0 && (ACCUM) != 0 && SXX_ABS(ELEM) > (1u<<20)*SXX_ABS(ACCUM)) { \
            TMP = ACCUM;                                                \
            ACCUM = CARRY;                                                          \
            CARRY = TMP;                                                            \
        }                                                                           \
    }                                                                               \
    TMP = (ACCUM) + (ELEM);

#if SXX_NEUMAIER_BRANCH == 1
#define SXX_NEUMAIER_ADD(ACCUM, CARRY, ELEM, TYPE, TMP, DO_SWAP) \
    do {                                                              \
        SXX_SWP_TMP_(ACCUM, CARRY, ELEM, TYPE, TMP, DO_SWAP)     \
        if (SXX_ABS(TMP) > SXX_ABS(ELEM)) {                 \
            CARRY += TYPE { (ACCUM) - (TMP) } + (ELEM);               \
        } else {                                                      \
            CARRY += TYPE { (ELEM) - (TMP) } + (ACCUM);               \
        }                                                             \
        ACCUM = (TMP);                                                \
    } while (0)
#else
#define SXX_NEUMAIER_ADD(ACCUM, CARRY, ELEM, TYPE, TMP, DO_SWAP)   \
    do {                                                                \
        SXX_SWP_TMP_(ACCUM, CARRY, ELEM, TYPE, TMP, DO_SWAP)       \
        T SXX_dat_[2] = {                                          \
            T { (ELEM) - (TMP) } + (ACCUM),                             \
            T { (ACCUM) - (TMP) } + (ELEM)                              \
        };                                                              \
        CARRY += SXX_dat_[SXX_ABS(TMP) > SXX_ABS(ELEM)]; \
        ACCUM = (TMP);                                                  \
    } while (0)
#endif
#undef SXX_CXX_SWP_TMP_
#define SXX_NEUMAIER_FINALIZE(ACCUM, CARRY) ((ACCUM) + (CARRY))
