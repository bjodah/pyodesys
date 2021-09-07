#pragma once
#ifndef SMMTNCXX_RESTRICT
#if defined(__GNUC__)
#define SMMTNCXX_RESTRICT __restrict__
#elif defined(_MSC_VER) && _MSC_VER >= 1400
#define SMMTNCXX_RESTRICT __restrict
// #elif defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
//   #define SMMTNCXX_RESTRICT restrict
#else
#define SMMTNCXX_RESTRICT
#endif
#endif

#ifndef SMMTNCXX_PREFER_INLINE
#if defined(__GNUC__)
#define SMMTNCXX_PREFER_INLINE __attribute__((flatten))
#elif defined(_MSC_VER) && _MSC_VER >= 1400
#define SMMTNCXX_PREFER_INLINE __forceinline
#else
#define SMMTNCXX_PREFER_INLINE
#endif
#endif

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
#define SMMTNCXX_FWD_IT_CONCEPT std::forward_iterator
#define SMMTNCXX_RND_IT_CONCEPT std::random_access_iterator
#else
#define SMMTNCXX_FWD_IT_CONCEPT typename
#define SMMTNCXX_RND_IT_CONCEPT typename
#endif

// Math macros to support e.g. __float128 without std lib support:
#ifndef SMMTNCXX_ABS
#define SMMTNCXX_ABS(x) (((x) < 0) ? -(x) : (x))
#endif

#ifndef SMMTNCXX_NEUMAIER_BRANCH
// see test/bench.cpp
#define SMMTNCXX_NEUMAIER_BRANCH 1
#endif

#if defined(SMMTNCXX_NEUMAIER_SWAP)
#error "API has changed, update your compilation flags accordingly"
#endif

#define SMMTNCXX_SWP_TMP_(ACCUM, CARRY, ELEM, TYPE, TMP, DO_SWAP)                   \
    TYPE TMP;                                                                       \
    if (DO_SWAP) {                                                                  \
        if (CARRY == 0 && ACCUM != 0 && SMMTNCXX_ABS(ELEM) > SMMTNCXX_ABS(ACCUM)) { \
            TMP = ACCUM;                                                            \
            ACCUM = CARRY;                                                          \
            CARRY = TMP;                                                            \
        }                                                                           \
    }                                                                               \
    TMP = (ACCUM) + (ELEM);

#if SMMTNCXX_NEUMAIER_BRANCH == 1
#define SMMTNCXX_NEUMAIER_ADD(ACCUM, CARRY, ELEM, TYPE, TMP, DO_SWAP) \
    do {                                                              \
        SMMTNCXX_SWP_TMP_(ACCUM, CARRY, ELEM, TYPE, TMP, DO_SWAP)     \
        if (SMMTNCXX_ABS(TMP) > SMMTNCXX_ABS(ELEM)) {                 \
            CARRY += TYPE { (ACCUM) - (TMP) } + (ELEM);               \
        } else {                                                      \
            CARRY += TYPE { (ELEM) - (TMP) } + (ACCUM);               \
        }                                                             \
        ACCUM = (TMP);                                                \
    } while (0)
#else
#define SMMTNCXX_NEUMAIER_ADD(ACCUM, CARRY, ELEM, TYPE, TMP, DO_SWAP)   \
    do {                                                                \
        SMMTNCXX_SWP_TMP_(ACCUM, CARRY, ELEM, TYPE, TMP, DO_SWAP)       \
        T SMMTNCXX_dat_[2] = {                                          \
            T { (ELEM) - (TMP) } + (ACCUM),                             \
            T { (ACCUM) - (TMP) } + (ELEM)                              \
        };                                                              \
        CARRY += SMMTNCXX_dat_[SMMTNCXX_ABS(TMP) > SMMTNCXX_ABS(ELEM)]; \
        ACCUM = (TMP);                                                  \
    } while (0)
#endif
#undef SMMTNCXX_CXX_SWP_TMP_
#define SMMTNCXX_NEUMAIER_FINALIZE(ACCUM, CARRY) ((ACCUM) + (CARRY))
