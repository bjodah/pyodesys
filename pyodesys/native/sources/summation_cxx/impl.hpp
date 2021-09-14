#pragma once
#include <array>
#include <cassert>
#include <summation_cxx/compensated.hpp>

namespace summation_cxx {
template <typename T, summation_cxx::Compensation Scheme>
struct Accumulator;
template <typename T, summation_cxx::Compensation Scheme>
struct AccuView;
namespace detail {
    template <typename T, Compensation scheme, typename Derived>
    struct Operators;
}
}
namespace summation_cxx::detail {
template <typename T, Compensation scheme, typename Derived>
struct Operators {
    typedef T underlying_type;
    typedef Accumulator<T, scheme> accumulator_type;
    typedef AccuView<T, scheme> view_type;

#define ACCUM(cv_qual) static_cast<cv_qual Derived*>(this)->accum()
#define CARRY(cv_qual) static_cast<cv_qual Derived*>(this)->carry()
    template <typename U>
    U to() const
    {
        if constexpr (Derived::compensation_scheme == Compensation::KAHAN) {
            return ACCUM(const);
        } else if constexpr (Derived::compensation_scheme == Compensation::NEUMAIER || Derived::compensation_scheme == Compensation::NEUMAIER_SWAP) {
            if constexpr (sizeof(T) > sizeof(U)) {
                return ACCUM(const) + CARRY(const);
            } else {
                return static_cast<U>(ACCUM(const)) + static_cast<U>(CARRY(const));
            }
        } else {
            assert(false);
        }
        return U { 0 } / U { 0 }; /* unreachable code, but would return NaN */
    }
    Derived& operator+=(T arg)
    {
        if constexpr (scheme == Compensation::KAHAN) {
            accum_kahan_destructive(ACCUM(), CARRY(), arg);
        } else if constexpr (scheme == Compensation::NEUMAIER) {
            accum_neumaier(ACCUM(), CARRY(), arg);
        } else if constexpr (scheme == Compensation::NEUMAIER_SWAP) {
            accum_neumaier_swap(ACCUM(), CARRY(), arg);
        } else {
            assert(false);
        }
        return *(static_cast<Derived*>(this));
    }
    Derived& operator-=(T arg)
    {
        Derived& self = *(static_cast<Derived*>(this));
        self += -arg;
        return self;
    }

    void operator=(const T& arg)
    {
        ACCUM() = arg;
        CARRY() = 0;
    }
    void operator/=(const T& arg)
    {
        ACCUM() /= arg;
        CARRY() /= arg;
    }
    void operator*=(const T& arg)
    {
        ACCUM() *= arg;
        CARRY() *= arg;
    }
    void operator+=(const accumulator_type& other)
    {
        *this += other.accum();
        CARRY() += other.carry();
    }
    void operator-=(const accumulator_type& other)
    {
        *this -= other.accum();
        CARRY() -= other.carry();
    }
    accumulator_type operator*(const T& arg) const
    {
        return accumulator_type(ACCUM(const) * arg, CARRY(const) * arg);
    }
    accumulator_type operator*(const accumulator_type& other) const
    {
        return accumulator_type(ACCUM(const) * other.accum(),
            CARRY(const) * other.accum() + ACCUM(const) * other.carry() + CARRY(const) * other.carry());
    }
    accumulator_type operator/(const accumulator_type& other) const
    {
        const T denom = other.template to<T>();
        return accumulator_type { ACCUM(const) / denom, CARRY(const) / denom };
    }
    accumulator_type operator+(const accumulator_type& other) const
    {
        return accumulator_type(ACCUM(const) + other.accum(), CARRY(const) + other.carry());
    }
    accumulator_type operator-(const accumulator_type& other) const
    {
        return accumulator_type(ACCUM(const) - other.accum(), CARRY(const) - other.carry());
    }
    accumulator_type operator+() const
    {
        return accumulator_type(ACCUM(const), CARRY(const));
    }
    accumulator_type operator-() const
    {
        return accumulator_type(-ACCUM(const), -CARRY(const));
    }
};
#define SMMTNCXX_COMMUTATIVE_OP(OP)                                           \
    template <typename Derived>                                               \
    typename Derived::accumulator_type operator OP(                           \
        const typename Derived::underlying_type& arg_a, const Derived& arg_b) \
    {                                                                         \
        return arg_b OP arg_a; /* multiplication is commutative */            \
    }
SMMTNCXX_COMMUTATIVE_OP(*)
SMMTNCXX_COMMUTATIVE_OP(+)
#define SMMTNCXX_PROMOTING_OP(OP)                                              \
    template <typename Derived>                                                \
    typename Derived::accumulator_type operator OP(                            \
        const typename Derived::underlying_type& arg_a, const Derived& arg_b)  \
    {                                                                          \
        return Derived { arg_a } OP arg_b; /* multiplication is commutative */ \
    }
SMMTNCXX_PROMOTING_OP(-)

#undef SMMTNCXX_COMMUTATIVE_OP
#undef SMMTNCXX_PROMOTING_OP
#undef ACCUM
#undef CARRY

}
