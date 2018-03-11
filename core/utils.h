#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <limits>
#include <algorithm>
#include <type_traits>

namespace utils {

template <typename T>
struct value_traits{
  using value_type = typename std::enable_if<std::is_arithmetic<T>::value, T>::type;
  using rational_type = typename std::conditional<std::is_floating_point<T>::value, T, double>::type;
};

template <typename T, std::enable_if_t<std::is_floating_point<T>::value, std::nullptr_t> = nullptr>
bool is_abs_close(T a, T b, T abs_tol = T(1e-10)) {
  return std::fabs(a - b) < abs_tol;
}

template <typename T, std::enable_if_t<std::is_floating_point<T>::value, std::nullptr_t> = nullptr>
bool is_close(T a, T b, T abs_tol = T(1e-10), T rel_tol = T(1e-10)) {
  auto diff = std::fabs(a - b);
  if (diff < abs_tol) {
    return true;
  }
  auto max_abs = std::max(std::fabs(a), std::fabs(b));
  if (diff < rel_tol * max_abs) {
    return true;
  }
  return false;
}

template <typename T, std::enable_if_t<std::is_integral<T>::value, std::nullptr_t> = nullptr>
bool is_close(T a, T b, T abs_tol = 0, T rel_tol = 0) {
  return a == b;
}

}

#endif
