#pragma once
#include <exception>
#include <mutex>

namespace anyode_parallel {
    class ThreadException {
        std::exception_ptr m_exc;
        std::mutex m_lock;
    public:
        ThreadException(): m_exc(nullptr) {}
        void rethrow() {
            if (m_exc) std::rethrow_exception(m_exc);
        }
        void capture_exception(){
            std::unique_lock<std::mutex> guard(m_lock);
            m_exc = std::current_exception();
        }
        template <typename F, typename... P>
        void run(F f, P... p){
            try {
                f(p...);
            } catch (...) {
                capture_exception();
            }
        }
        bool holds_exception() { return m_exc != nullptr; }
    };
}
