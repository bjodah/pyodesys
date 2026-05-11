#pragma once
#include "summation_cxx/compensated.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <summation_cxx/impl.hpp>

namespace summation_cxx {

template <typename T, Compensation scheme>
struct Accumulator : public detail::Operators<T, scheme, Accumulator<T, scheme>> {
    static constexpr Compensation compensation_scheme { scheme };

private:
    std::array<T, 2> data {};

public:
    T& accum() { return data.data()[0]; }
    T& carry() { return data.data()[1]; }
    const T& accum() const { return data.data()[0]; }
    const T& carry() const { return data.data()[1]; }

public:
    Accumulator() = default;
    // cppcheck-suppress noExplicitConstructor
    Accumulator(T accum)
    {
        data[0] = accum;
    }
    explicit Accumulator(T accum, T carry)
    {
        data[0] = accum;
        data[1] = carry;
    }
    void clear() { data.clear(); }

    template <int n>
    static constexpr T sum(const std::array<T, n>& arr)
    {
        Accumulator ta {};
#if defined(__clang__)
#pragma unroll 16
#elif defined(__GNUC__)
#pragma GCC unroll 16
#endif
        for (const auto& e : arr) {
            // cppcheck-suppress useStlAlgorithm
            ta += e;
        }
        return ta.template to<T>();
    }

private:
    template <typename U, typename... Us>
    static constexpr void from_(Accumulator<T, scheme>& acu, U arg, Us... args)
    {
        acu += arg;
        if constexpr (sizeof...(args) > 0) {
            from_(acu, args...);
        }
    }

public:
    template <typename U, typename... Us>
    static constexpr Accumulator<T, scheme> from(U arg, Us... args)
    {
        Accumulator<T, scheme> acu { arg };
        if constexpr (sizeof...(args) > 0) {
            Accumulator<T, scheme>::from_(acu, args...);
        }
        return acu;
    }
};
template <typename T>
using AccumulatorKahan = Accumulator<T, Compensation::KAHAN>;
template <typename T>
using AccumulatorNeumaier = Accumulator<T, Compensation::NEUMAIER>;
template <typename T>
using AccumulatorNeumaierSwap = Accumulator<T, Compensation::NEUMAIER_SWAP>;
template <typename T>
using AccumulatorTwoSum = Accumulator<T, Compensation::TWO_SUM>;


template <typename T, Compensation scheme>
T pow(const Accumulator<T, scheme>& base, T exponent)
{
    return std::pow(base.accum(), exponent);
}
template <typename T, Compensation scheme>
T pow(const Accumulator<T, scheme>& base, int exponent)
{
    return std::pow(base.accum(), static_cast<T>(exponent));
}

}
