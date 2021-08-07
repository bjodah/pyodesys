// -*- eval: (read-only-mode); -*-
#pragma once
#include <array>

namespace {
template <typename T, typename... Ts>
void add_neumaier_rvt_(std::array<T, 2> &work, T arg, Ts... args) {
  do {
    T tmp = (work[0]) + (arg);
    if ((((tmp) < 0) ? -(tmp) : (tmp)) > (((arg) < 0) ? -(arg) : (arg))) {
      work[1] += T{(work[0]) - (tmp)} + (arg);
    } else {
      work[1] += T{(arg) - (tmp)} + (work[0]);
    }
    work[0] = (tmp);
  } while (0);
  if constexpr (sizeof...(args) > 0) {
    add_neumaier_rvt_(work, args...);
  }
}
} // namespace

template <typename T, typename... Ts> T add_neumaier_rvt(T arg, Ts... args) {
  std::array<T, 2> work{};
  do {
    T tmp = (work[0]) + (arg);
    if ((((tmp) < 0) ? -(tmp) : (tmp)) > (((arg) < 0) ? -(arg) : (arg))) {
      work[1] += T{(work[0]) - (tmp)} + (arg);
    } else {
      work[1] += T{(arg) - (tmp)} + (work[0]);
    }
    work[0] = (tmp);
  } while (0);
  add_neumaier_rvt_(work, args...);
  return ((work[0]) + (work[1]));
}
