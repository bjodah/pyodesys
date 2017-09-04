#pragma once

#ifdef NDEBUG
#include<memory>
#else
#include<vector>
#endif

namespace AnyODE {

#ifdef NDEBUG
#if __cplusplus >= 201402L
    using std::make_unique;
#else
    template <class T, class ...Args>
    typename std::enable_if
    <
        !std::is_array<T>::value,
        std::unique_ptr<T>
        >::type
    make_unique(Args&& ...args)
    {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }

    template <class T>
    typename std::enable_if
    <
        std::is_array<T>::value,
        std::unique_ptr<T>
        >::type
    make_unique(std::size_t n)
    {
        typedef typename std::remove_extent<T>::type RT;
        return std::unique_ptr<T>(new RT[n]);
    }
#endif

    template<typename T> using buffer_t = std::unique_ptr<T[]>;
    template<typename T> using buffer_ptr_t = T*;
    template<typename T> constexpr T* buffer_get_raw_ptr(buffer_t<T>& buf) {
        return buf.get();
    }
    template<typename T> inline constexpr buffer_t<T> buffer_factory(std::size_t n) {
        return make_unique<T[]>(n);
    }

#else
    template<typename T> using buffer_t = std::vector<T>;
    template<typename T> using buffer_ptr_t = T*;
    template<typename T> inline constexpr buffer_t<T> buffer_factory(std::size_t n) {
        return buffer_t<T>(n);
    }
    template<typename T> constexpr T* buffer_get_raw_ptr(buffer_t<T>& buf) {
        return &buf[0];
    }
#endif
}
