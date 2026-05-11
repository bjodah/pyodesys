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
        } else if constexpr (Derived::compensation_scheme == Compensation::NEUMAIER
                             || Derived::compensation_scheme == Compensation::NEUMAIER_SWAP
                             || Derived::compensation_scheme == Compensation::TWO_SUM
                             || Derived::compensation_scheme == Compensation::FAST_TWO_SUM
            ) {
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
        } else if constexpr (scheme == Compensation::TWO_SUM) {
            accum_two_sum(ACCUM(), CARRY(), arg);
        } else if constexpr (scheme == Compensation::FAST_TWO_SUM) {
            accum_fast_two_sum(ACCUM(), CARRY(), arg);
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

    Derived& operator=(const T arg)
    {
        Derived& self = *(static_cast<Derived *>(this));
        ACCUM() = arg;
        CARRY() = 0;
        return self;
    }
    void operator/=(const T& arg)
    {
        ACCUM() /= arg;
        CARRY() /= arg;
    }
    void operator*=(const T& arg)
    {
        const T ori {ACCUM()};
        ACCUM() *= arg;
        CARRY() *= arg;
        CARRY() += fma(ori, arg, -ACCUM()); // 2product
    }
    Derived& operator+=(const accumulator_type& other)
    {
        Derived& self = *(static_cast<Derived*>(this));
        self += other.accum();
        self /*CARRY()*/ += other.carry();
        return self;
    }
    Derived& operator-=(const accumulator_type& other)
    {
        Derived& self = *(static_cast<Derived*>(this));
        self -= other.accum();
        self /*CARRY()*/ -= other.carry();
        return self;
    }
    accumulator_type operator*(const T& arg) const
    {
        Derived cpy = *(static_cast<const Derived*>(this));
        cpy *= arg;
        return cpy;
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
        Derived cpy = *(static_cast<const Derived*>(this));
        cpy += other;
        return cpy;
    }
    accumulator_type operator+(const T& arg) const
    {
        Derived cpy = *(static_cast<const Derived*>(this));
        cpy += arg;
        return cpy;
    }
    accumulator_type operator-(const accumulator_type& other) const
    {
        Derived cpy = *(static_cast<const Derived*>(this));
        cpy -= other;
        return cpy;
    }
    accumulator_type operator+() const
    {
        return accumulator_type(ACCUM(const), CARRY(const));
    }
    accumulator_type operator-() const
    {
        return accumulator_type(-ACCUM(const), -CARRY(const));
    }
#define SXX_COMP(OPER_)                                                 \
    bool operator OPER_(const T& arg) const                             \
    {                                                                   \
        const Derived& self = *(static_cast<const Derived*>(this));     \
        return self.template to<T>() OPER_ arg;                         \
    }
    SXX_COMP(<)
    SXX_COMP(>)
};
#undef SXX_COMP
#define SXX_COMMUTATIVE_OP(OP)                                           \
    template <typename Derived>                                               \
    typename Derived::accumulator_type operator OP(                           \
        const typename Derived::underlying_type& arg_a, const Derived& arg_b) \
    {                                                                         \
        return arg_b OP arg_a; /* multiplication is commutative */            \
    }
SXX_COMMUTATIVE_OP(*)
SXX_COMMUTATIVE_OP(+)
#define SXX_PROMOTING_OP(OP)                                              \
    template <typename Derived>                                                \
    typename Derived::accumulator_type operator OP(                            \
        const typename Derived::underlying_type& arg_a, const Derived& arg_b)  \
    {                                                                          \
        return Derived { arg_a } OP arg_b; /* multiplication is commutative */ \
    }
SXX_PROMOTING_OP(-)

#undef SXX_COMMUTATIVE_OP
#undef SXX_PROMOTING_OP
#undef ACCUM
#undef CARRY

}
