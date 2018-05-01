#include <vector>

#if !defined(BEGIN_NAMESPACE)
#define BEGIN_NAMESPACE(s) namespace s{
#endif
#if !defined(END_NAMESPACE)
#define END_NAMESPACE(s) }
#endif

BEGIN_NAMESPACE(AnyODE)
template<typename T>
void extend_vec(std::vector<T> &dest, const std::vector<T> &source){
    dest.reserve(dest.size() + std::distance(source.begin(), source.end()));
    dest.insert(dest.end(), source.begin(), source.end());
}
END_NAMESPACE(AnyODE)
