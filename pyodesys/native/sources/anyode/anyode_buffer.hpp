#pragma once

#ifdef NDEBUG
#include<memory>
#else
#include<vector>
#endif

namespace AnyODE {

#ifdef NDEBUG
    template<typename T> using buffer_t = std::unique_ptr<T[]>;
    template<typename T> using buffer_ptr_t = T*;
    template<typename T> constexpr T* buffer_get_raw_ptr(buffer_t<T>& buf) {
        return buf.get();
    }
    template<typename T> inline constexpr buffer_t<T> buffer_factory(std::size_t n) {
        return std::make_unique<T[]>(n);
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
