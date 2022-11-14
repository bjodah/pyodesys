#pragma once
#ifdef __FAST_MATH__
#error fast math enabled (/fp:fast, -ffast-math), this would negate compensation.
#endif
#include "summation_cxx/macros.hpp"
#include <cstddef> // std::size_t

namespace summation_cxx {
enum class Compensation {
    NONE,
    KAHAN, // should be equivalent to FAST_TWO_SUM
    NEUMAIER,
    NEUMAIER_SWAP,
    TWO_SUM,
    FAST_TWO_SUM
};

namespace /* anonymous */ {
    template <typename T>
    SXX_PREFER_INLINE void accum_kahan_destructive(
        T& SXX_RESTRICT accu,
        T& SXX_RESTRICT carry,
        T& SXX_RESTRICT elem)
    {
        elem -= carry;
        const T tmp = accu + elem;
        carry = T { tmp - accu } - elem;
        accu = tmp;
    }
    template <typename T>
    SXX_PREFER_INLINE void accum_kahan(
        T& SXX_RESTRICT accu,
        T& SXX_RESTRICT carry,
        const T& SXX_RESTRICT elem)
    {
        T y = elem;
        accum_kahan_destructive(accu, carry, y);
    }

    template <typename T>
    SXX_PREFER_INLINE void accum_neumaier(
        T& SXX_RESTRICT acm,
        T& SXX_RESTRICT carry,
        const T& SXX_RESTRICT elem)
    {
        SXX_NEUMAIER_ADD(acm, carry, elem, T, tmp, false);
    }

    template <typename T>
    SXX_PREFER_INLINE void accum_neumaier_swap(
        T& SXX_RESTRICT acm,
        T& SXX_RESTRICT carry,
        const T& SXX_RESTRICT elem)
    {
        // cppcheck-suppress redundantAssignment
        SXX_NEUMAIER_ADD(acm, carry, elem, T, tmp, true);
    }

    template <typename T>
    SXX_PREFER_INLINE void accum_two_sum(
        T& SXX_RESTRICT accu,
        T& SXX_RESTRICT carry,
        const T& SXX_RESTRICT elem)
    {
        const T s = accu + elem;
        const T ap = s - elem;
        const T bp = s - ap;
        const T da = accu - ap;
        const T db = elem - bp;
        carry += da + db;
        accu = s;
    }

    template <typename T>
    SXX_PREFER_INLINE void accum_fast_two_sum(
        T& SXX_RESTRICT accu,
        T& SXX_RESTRICT carry,
        const T& SXX_RESTRICT elem)
    {
        const T s = accu + elem;
        const T z = s - accu;
        const T t = elem - z;
        carry += t;
        accu = s;
    }
}
}
