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

#ifndef ORACLES_IWATA_TEST_FUNCTION_H
#define ORACLES_IWATA_TEST_FUNCTION_H

#include <vector>
#include <stdexcept>
//#include <numeric>
#include "core/oracle.h"
#include "core/set_utils.h"

namespace submodular {

template <typename ValueType>
class IwataTestFunction: public SubmodularOracle<ValueType> {
public:
  using value_type = typename ValueTraits<ValueType>::value_type;

  IwataTestFunction(std::size_t n): n_(n) { this->SetDomain(Set::MakeDense(n_)); }

  value_type Call(const Set& X) {
    if (X.n_ != n_) {
      throw std::range_error("IwataTestFunction::Call: Input size mismatch");
    }
    auto elements = X.GetMembers();
    auto card = value_type(elements.size());
    auto sum = value_type(0);
    auto n = value_type(n_);
    for (const auto& i: elements) {
      sum += value_type(i + 1);
    }
    return card * (n - card) + (2 * card * n) - (5 * sum);
  }

  std::string GetName() { return "Iwata test function"; }

  std::size_t GetN() const { return n_; }
  std::size_t GetNGround() const { return n_; }

private:
  std::size_t n_;

};

template <typename ValueType>
class GroupwiseIwataTestFunction: public SubmodularOracle<ValueType> {
public:
  using value_type = typename ValueTraits<ValueType>::value_type;

  GroupwiseIwataTestFunction(std::size_t n, std::size_t k)
    : n_(n), k_(k)
  {
    this->SetDomain(Set::MakeDense(n_));

    if (k_ < 1) k_ = 1;
    if (k_ > n_) k_ = n_;
  }

  value_type Call(const Set& X) {
    if (X.n_ != n_) {
      throw std::range_error("GroupwiseIwataTestFunction::Call: Input size mismatch");
    }

    auto elements = X.GetMembers();
    std::vector<value_type> groupwise_card(k_, 0);
    std::vector<value_type> groupwise_sum(k_, 0);
    auto val = value_type(0);

    for (const auto& i: elements) {
      groupwise_card[i % k_] += static_cast<value_type>(1);
      groupwise_sum[i % k_] += static_cast<value_type>((i / k_) + 1);
    }

    for (std::size_t j = 0; j < k_; ++j) {
      auto nj = static_cast<value_type>(n_ / k_);
      nj += static_cast<value_type>(j < (n_ % k_) ? 1 : 0);
      val += groupwise_card[j] * (nj - groupwise_card[j]) + (2 * groupwise_card[j] * nj) - (5 * groupwise_sum[j]);
    }
    return val;
  }

  std::string GetName() { return "Groupwise Iwata test function"; }

  std::size_t GetN() const { return n_; }
  std::size_t GetNGround() const { return n_; }

private:
  std::size_t n_;
  std::size_t k_; // number of groups
};

}

#endif
