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

#ifndef ORACLES_MODULAR_H
#define ORACLES_MODULAR_H

#include <vector>
#include <stdexcept>
#include "core/oracle.h"
#include "core/set_utils.h"

namespace submodular {

template <typename ValueType>
class ModularOracle: public SubmodularOracle<ValueType> {
public:
  using value_type = typename ValueTraits<ValueType>::value_type;

  ModularOracle(const std::vector<value_type>& x)
    :n_(x.size()), x_(x) { this->SetDomain(Set::MakeDense(n_)); }
  ModularOracle(std::vector<value_type>&& x)
    :n_(x.size()), x_(std::move(x)) { this->SetDomain(Set::MakeDense(n_)); }

  value_type Call(const Set& X) {
    if (X.n_ != n_) {
      throw std::range_error("ModularOracle::Call: Input size mismatch");
    }
    value_type f(0);
    for (auto& i: X.GetMembers()) {
      f += x_[i];
    }
    return f;
  }

  std::string GetName() { return "Modular"; }

  std::size_t GetN() const { return n_; }
  std::size_t GetNGround() const { return n_; }

private:
  std::size_t n_;
  std::vector<value_type> x_;
};

template <typename ValueType>
class ConstantOracle: public SubmodularOracle<ValueType> {
public:
  using value_type = typename ValueTraits<ValueType>::value_type;

  ConstantOracle(std::size_t n, value_type c): n_(n), c_(c) { this->SetDomain(Set::MakeDense(n_)); }

  value_type Call(const Set& X) {
    if (X.n_ != n_) {
      throw std::range_error("ConstantOracle::Call: Input size mismatch");
    }
    return c_;
  }

  std::string GetName() { return "Constant"; }

  std::size_t GetN() const { return n_; }
  std::size_t GetNGround() const { return n_; }

private:
  std::size_t n_;
  value_type c_;
};


}

#endif
