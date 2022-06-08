#pragma once
#include <cstddef>
#include <cstring> // std::memset
#include <memory> // std::make_unique
#include <summation_cxx/accumulator.hpp>
#include <summation_cxx/compensated.hpp>
#include <summation_cxx/view.hpp>

namespace summation_cxx {
template <typename T, Compensation scheme, typename U = void /* target_type */>
struct RangedAccumulator {
    typedef T underlying_type;
    typedef std::conditional_t<std::is_same_v<U, void>, T, U> target_type;
    typedef Accumulator<T, scheme> accumulator_type;
    typedef AccuView<T, scheme> view_type;

private:
    target_type* tgt {};
    std::unique_ptr<underlying_type[]> storage {};
    std::size_t sz {};
    bool cumulative {};

public:
    RangedAccumulator() = default;
    RangedAccumulator(std::size_t sz)
        : storage(std::make_unique<underlying_type[]>(sz * 2))
        , sz(sz)
    {
    }
    void init(target_type* target, bool cumulative = false)
    {
        tgt = target;
        this->cumulative = cumulative;
        if (sz > 0 /* UB to call memset over zero bytes. */) {
            // doing this only makes sense if commit() is not always called.
            std::memset(storage.get(), 0x00, sizeof(underlying_type) * sz * 2);
        }
    }
    view_type operator[](std::size_t idx)
    {
        return view_type { &storage[idx * 2] };
    }
    const view_type operator[](std::size_t idx) const
    {
        return view_type { &storage[idx * 2] };
    }

    void commit() const
    {
#if defined(SUMMTNCXX_DISTRUST_OPTIMIZING_COMPILERS)
#define SXX_OUTPUT(OP)                        \
    if constexpr (scheme == Compensation::KAHAN) { \
        this->tgt[i] OP this->storage[i * 2];      \
    } else if constexpr (scheme == Compensation::NEUMAIER  ||
        Derived::compensation_scheme == Compensation::NEUMAIER_SWAP)
        {
            this->tgt[i] OP this->storage[i * 2] + this->storage[i * 2 + 1];
        }
        else
        {
            assert(false);
        }
#else
#define SXX_OUTPUT(OP) this->tgt[i] OP(*this)[i].template to<target_type>();
#endif
#define SXX_LOOP  \
    std::size_t i = 0; \
    i < this->sz;      \
    ++i
        if (cumulative) {
            for (SXX_LOOP) {
                SXX_OUTPUT(+=)
            }
        } else {
            for (SXX_LOOP) {
                SXX_OUTPUT(=)
            }
        }
#undef SXX_LOOP
#undef SXX_OUTPUT
    }
};
template <typename T, typename U = void>
using RangedAccumulatorKahan = RangedAccumulator<T, Compensation::KAHAN, U>;
template <typename T, typename U = void>
using RangedAccumulatorNeumaier = RangedAccumulator<T, Compensation::NEUMAIER, U>;
template <typename T, typename U = void>
using RangedAccumulatorNeumaierSwap = RangedAccumulator<T, Compensation::NEUMAIER_SWAP, U>;

/// Simplifies writing generic code against ranged.hpp, no compensation:
template <typename T, typename U = void>
struct RangedUncompensatedView {
    typedef T underlying_type;
    typedef std::conditional_t<std::is_same_v<U, void>, T, U> target_type;
    typedef T accumulator_type;
    typedef target_type& view_type;

protected:
    target_type* tgt {};
    std::size_t sz {};

public:
    RangedUncompensatedView() = default;
    RangedUncompensatedView(std::size_t sz)
        : sz(sz)
    {
    }
    void init(target_type* target, bool cumulative = false)
    {
        tgt = target;
        if (!cumulative && (sz > 0 /* UB to call memset over zero bytes. */)) {
            // may be skipped e.g. if we know target is already zero-initialized
            std::memset(target, 0x00, sizeof(target_type) * sz);
        }
    }
    view_type operator[](std::size_t idx)
    {
        return tgt[idx];
    }
    void commit() const { } // no-op
};
}
