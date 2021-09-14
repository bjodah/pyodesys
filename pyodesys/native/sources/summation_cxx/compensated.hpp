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
    SMMTNCXX_PREFER_INLINE void accum_kahan_destructive(
        T& SMMTNCXX_RESTRICT accu,
        T& SMMTNCXX_RESTRICT carry,
        T& SMMTNCXX_RESTRICT elem)
    {
        elem -= carry;
        const T tmp = accu + elem;
        carry = T { tmp - accu } - elem;
        accu = tmp;
    }
    template <typename T>
    SMMTNCXX_PREFER_INLINE void accum_kahan(
        T& SMMTNCXX_RESTRICT accu,
        T& SMMTNCXX_RESTRICT carry,
        const T& SMMTNCXX_RESTRICT elem)
    {
        T y = elem;
        accum_kahan_destructive(accu, carry, y);
    }

    template <typename T>
    SMMTNCXX_PREFER_INLINE void accum_neumaier(
        T& SMMTNCXX_RESTRICT acm,
        T& SMMTNCXX_RESTRICT carry,
        const T& SMMTNCXX_RESTRICT elem)
    {
        SMMTNCXX_NEUMAIER_ADD(acm, carry, elem, T, tmp, false);
    }

    template <typename T>
    SMMTNCXX_PREFER_INLINE void accum_neumaier_swap(
        T& SMMTNCXX_RESTRICT acm,
        T& SMMTNCXX_RESTRICT carry,
        const T& SMMTNCXX_RESTRICT elem)
    {
        // cppcheck-suppress redundantAssignment
        SMMTNCXX_NEUMAIER_ADD(acm, carry, elem, T, tmp, true);
    }
}
}
