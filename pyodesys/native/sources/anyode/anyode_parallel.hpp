#ifndef ANYODE_PARALLEL_H_47FBBF5A914C11E6AAE25B5C5A8B7CFC
#define ANYODE_PARALLEL_H_47FBBF5A914C11E6AAE25B5C5A8B7CFC

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
    };
}

#endif /* ANYODE_PARALLEL_H_47FBBF5A914C11E6AAE25B5C5A8B7CFC */
