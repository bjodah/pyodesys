#pragma once
#include <summation_cxx/impl.hpp>

namespace summation_cxx {

template <typename T, Compensation scheme>
struct AccuView : public detail::Operators<T, scheme, AccuView<T, scheme>> {
    static constexpr Compensation compensation_scheme { scheme };

private:
    T* ptr;

public:
    T& accum() { return ptr[0]; }
    T& carry() { return ptr[1]; }
    const T& accum() const { return ptr[0]; }
    const T& carry() const { return ptr[1]; }

public:
    AccuView() = delete;
    AccuView<T, scheme> & operator=(const AccuView<T, scheme>&) = delete;
    using detail::Operators<T, scheme, AccuView<T, scheme>>::operator=;
    // cppcheck-suppress noExplicitConstructor
    AccuView(T* data)
        : ptr(data)
    {
        assert(data);
    }
    Accumulator<T, scheme> deepcopy()
    {
        return Accumulator<T, scheme> { this->accum(), this->carry() };
    }
};

template <typename T>
using AccuViewKahan = AccuView<T, Compensation::KAHAN>;
template <typename T>
using AccuViewNeumaier = AccuView<T, Compensation::NEUMAIER>;
template <typename T>
using AccuViewNeumaierSwap = AccuView<T, Compensation::NEUMAIER_SWAP>;
}
