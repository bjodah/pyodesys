#pragma once
#ifdef __FAST_MATH__
#error fast math enabled (/fp:fast, -ffast-math), this would negate compensation.
#endif
#include "summation_cxx/macros.hpp"
#include <cstddef> // std::size_t

namespace summation_cxx {
enum class Compensation { NONE,
    KAHAN,
    NEUMAIER,
    NEUMAIER_SWAP };

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
}
}
