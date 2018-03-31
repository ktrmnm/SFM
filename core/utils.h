// Copyright 2018 Kentaro Minami. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <limits>
#include <utility>
#include <algorithm>
//#include <memory>
#include <type_traits>
//#include <functional>
#include <iterator>
#include <unordered_map>
//#include <chrono>

namespace utils {

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
bool is_abs_close(T a, T b, T abs_tol = 0) {
  return a == b;
}

template <typename T, std::enable_if_t<std::is_integral<T>::value, std::nullptr_t> = nullptr>
bool is_close(T a, T b, T abs_tol = 0, T rel_tol = 0) {
  return a == b;
}

template <
  typename T,
  std::enable_if_t<std::is_floating_point<T>::value, std::nullptr_t> = nullptr
>
T abs_max(std::vector<T> x) {
  auto z = *std::max_element(x.begin(), x.end(),
                            [](const T& a, const T& b){ return std::fabs(a) < std::fabs(b); });
  return fabs(z);
}

template <
  typename T,
  std::enable_if_t<std::is_integral<T>::value, std::nullptr_t> = nullptr
>
double abs_max(std::vector<T> x) {
  auto z = *std::max_element(x.begin(), x.end(),
                            [](const T& a, const T& b){ return std::fabs(a) < std::fabs(b); });
  return fabs(z);
}


//template<typename Key, typename T,
//        typename Hash = std::hash<Key>, typename Pred = std::equal_to<Key>,
//        typename Allocator = std::allocator<std::pair<const Key, T>>>
template <typename Key, typename T>
class unordered_map_value_iterator
#if __cplusplus < 201500
  : std::iterator<
      typename std::iterator_traits<typename std::unordered_map<Key, T>::iterator>::iterator_category, T
    >
#endif
{
private:
  using orig_iterator_type = typename std::unordered_map<Key, T>::iterator;
  orig_iterator_type it_;
#if __cplusplus < 201500
  using base_type = std::iterator<
    typename std::iterator_traits<typename std::unordered_map<Key, T>::iterator>::iterator_category, T
  >;
#endif
public:
  unordered_map_value_iterator() = default;
  unordered_map_value_iterator(orig_iterator_type it) noexcept: it_(it) {}
  unordered_map_value_iterator(const unordered_map_value_iterator&) = default;
  unordered_map_value_iterator(unordered_map_value_iterator&&) = default;
  unordered_map_value_iterator& operator=(const unordered_map_value_iterator&) = default;
  unordered_map_value_iterator& operator=(unordered_map_value_iterator&&) = default;
  orig_iterator_type get_raw_iterator() const { return it_; }
#if __cplusplus < 201500
  using iterator_category = typename base_type::iterator_category;
  using value_type = typename base_type::value_type;
  using difference_type = typename base_type::difference_type;
  using pointer = typename base_type::pointer;
  using reference = typename base_type::reference;
#else
  using iterator_category = typename std::iterator_traits<orig_iterator_type>::iterator_category;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using reference = T&;
#endif
  T& operator*() noexcept { return it_->second; }
  T operator*() const noexcept { return it_->second; }
  T* operator->() noexcept { return &it_->second; }
  const T* operator->() const noexcept { return &it_->second; }
  unordered_map_value_iterator& operator++() noexcept {
    ++it_;
    return *this;
  }
  unordered_map_value_iterator operator++(int) noexcept {
    const auto retval = *this;
    ++*this;
    return retval;
  }
};

template <typename Key, typename T>
bool operator==(const unordered_map_value_iterator<Key, T>& lhs, const unordered_map_value_iterator<Key, T>& rhs) {
  return lhs.get_raw_iterator() == rhs.get_raw_iterator();
}
template <typename Key, typename T>
bool operator!=(const unordered_map_value_iterator<Key, T>& lhs, const unordered_map_value_iterator<Key, T>& rhs) {
  return !(lhs == rhs);
}

}

#endif
