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

#ifndef PARTIAL_VECTOR_H
#define PARTIAL_VECTOR_H

#include <vector>
#include <utility>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include "core/set_utils.h"

namespace submodular {

template <typename T> class PartialVector;

template <typename T>
PartialVector<T> operator+(const PartialVector<T>& x1, const PartialVector<T>& x2);

template <typename T>
PartialVector<T> operator*(T a, const PartialVector<T>& x1);

template <typename T>
std::ostream& operator << (std::ostream& stream, const PartialVector<T>& vec);

template <typename T>
class PartialVector {
public:
  PartialVector() = delete;
  PartialVector(const PartialVector&) = default;
  PartialVector(PartialVector&&) = default;
  PartialVector& operator=(const PartialVector&) = default;
  PartialVector& operator=(PartialVector&&) = default;

  explicit PartialVector(std::size_t n)
    :n_ground_(n), n_(n), domain_(Set::MakeDense(n)), x_(n, T(0)) {}
  explicit PartialVector(const std::vector<T>& x)
    :n_ground_(x.size()), n_(n_ground_), domain_(Set::MakeDense(n_ground_)), x_(x) {}
  explicit PartialVector(std::vector<T>&& x)
    :n_ground_(x.size()), n_(n_ground_), domain_(Set::MakeDense(n_ground_)), x_(std::move(x)) {}
  explicit PartialVector(const Set& domain)
    :n_ground_(domain.n_), n_(domain.Cardinality()), domain_(domain), x_(n_ground_, T(0)) {}
  explicit PartialVector(Set&& domain)
    :n_ground_(domain.n_), n_(domain.Cardinality()), domain_(std::move(domain)), x_(n_ground_, T(0)) {}
  PartialVector(const std::vector<T>& x, const Set& domain)
    : n_ground_(domain.n_), n_(domain.Cardinality()), domain_(domain), x_(x) {}
  PartialVector(std::vector<T>&& x, const Set& domain)
    : n_ground_(domain.n_), n_(domain.Cardinality()), domain_(domain), x_(std::move(x)) {}
  PartialVector(const std::vector<T>& x, Set&& domain)
    : n_ground_(domain.n_), n_(domain.Cardinality()), domain_(std::move(domain)), x_(x) {}
  PartialVector(std::vector<T>&& x, Set&& domain)
    : n_ground_(domain.n_), n_(domain.Cardinality()), domain_(std::move(domain)), x_(std::move(x)) {}

  // Subscript operator for partial vector.
  // If pos is in the domain, it returns the value of x_[pos].
  // Otherwise it throws std::range_error.
  const T& operator[](std::size_t pos) const noexcept(false);
  T& operator[](std::size_t pos) noexcept(false);

  PartialVector<T>& operator+=(const PartialVector<T>& x);
  PartialVector<T> operator-() const;
  //PartialVector<T> operator+() const;
  PartialVector<T> Multiply(T multiplier) const;

  std::vector<T> GetVector() const;
  Set GetDomain() const;

  // Get a "short" re-indexed vector whose length is same as the domain cardinality
  std::vector<T> GetActiveVector() const;
  // Set data from a sshort vector
  void SetActiveVector(const std::vector<T>& data);

  // Make sure that x_.size() == n_ground_, domain_.n_ == n_ground_, domain_.Cardinality() == n_
  // (PartialVector object does not check the consistency by itself.)
  std::size_t n_ground_;
  std::size_t n_;
  Set domain_;
  std::vector<T> x_;

  friend std::ostream& operator << (std::ostream&, const PartialVector&);
};

template <typename T>
const T& PartialVector<T>::operator[](std::size_t pos) const noexcept(false) {
  if (!domain_.HasElement(pos)) {
    throw std::range_error("PartialVector");
  }
  return x_[pos];
}

template <typename T>
T& PartialVector<T>::operator[](std::size_t pos) noexcept(false) {
  if (!domain_.HasElement(pos)) {
    throw std::range_error("PartialVector");
  }
  return x_[pos];
}

template <typename T>
PartialVector<T>& PartialVector<T>::operator+=(const PartialVector<T>& x) {
  auto domain_new = domain_.Union(x.domain_);
  for (const auto& i: domain_new.GetMembers()) {
    if (domain_[i] && x.domain_[i]) {
      x_[i] += x.x_[i];
    }
    else if (!domain_[i] && x.domain_[i]) {
      x_[i] = x.x_[i];
    }
  }
  return *this;
}

template <typename T>
PartialVector<T> PartialVector<T>::operator-() const {
  std::vector<T> x_new(n_ground_, T(0));
  Set domain_copy = domain_.Copy();
  for (std::size_t i = 0; i < n_ground_; ++i) {
    if (domain_[i]) {
      x_new[i] = - x_[i];
    }
  }
  return PartialVector<T>(std::move(x_new), std::move(domain_copy));
}

template <typename T>
PartialVector<T> PartialVector<T>::Multiply(T multiplier) const {
  std::vector<T> x_new(n_ground_, T(0));
  Set domain_copy = domain_.Copy();
  for (std::size_t i = 0; i < n_ground_; ++i) {
    if (domain_[i]) {
      x_new[i] = multiplier * x_[i];
    }
  }
  return PartialVector<T>(std::move(x_new), std::move(domain_copy));
}

template <typename T>
std::vector<T> PartialVector<T>::GetVector() const {
  std::vector<T> x_new(x_);
  return x_new;
}

template <typename T>
Set PartialVector<T>::GetDomain() const {
  return domain_.Copy();
}

template <typename T>
std::vector<T> PartialVector<T>::GetActiveVector() const {
  std::vector<T> data;
  for (const auto& i: domain_.GetMembers()) {
    data.push_back(x_[i]);
  }
  return data;
}

template <typename T>
void PartialVector<T>::SetActiveVector(const std::vector<T>& data) {
  auto members = domain_.GetMembers();
  auto m = std::min(data.size(), members.size());
  for (std::size_t i = 0; i < m; ++i) {
    x_[members[i]] = data[i];
  }
}

template <typename T>
PartialVector<T> operator+(const PartialVector<T>& x1, const PartialVector<T>& x2) {
  auto domain_new = x1.GetDomain().Union(x2.domain_);
  std::vector<T> x_new(x1.n_ground_);
  for (const auto& i: domain_new.GetMembers()) {
    if (x1.domain_[i] && x2.domain_[i]) {
      x_new[i] = x1.x_[i] + x2.x_[i];
    }
    else if (x1.domain_[i] && !x2.domain_[i]) {
      x_new[i] = x1.x_[i];
    }
    else if (!x1.domain_[i] && x2.domain_[i]) {
      x_new[i] = x2.x_[i];
    }
  }
  return PartialVector<T>(std::move(x_new), std::move(domain_new));
}

template <typename T>
PartialVector<T> operator*(T a, const PartialVector<T>& x1) {
  auto x_new = x1.GetVector();
  auto domain_new = x1.GetDomain();
  for (const auto& i: domain_new.GetMembers()) {
    x_new[i] *= a;
  }
  return PartialVector<T>(std::move(x_new), std::move(domain_new));
}

template <typename T>
std::ostream& operator << (std::ostream& stream, const PartialVector<T>& vec) {
  stream << "[";
  for (std::size_t i = 0; i < vec.n_ground_; ++i) {
    if (vec.domain_.HasElement(i)) {
      stream << vec.x_[i];
    }
    else {
      stream << "*";
    }
    if (i < vec.n_ground_ - 1) {
      stream << ", ";
    }
  }
  stream << "]";
  return stream;
}

}

#endif
