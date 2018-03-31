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


#ifndef ALGORITHMS_SFM_FW_H
#define ALGORITHMS_SFM_FW_H

#include <utility>
#include <vector>
#include <limits>
#include <algorithm>
#include <iostream>
#include "core/utils.h"
#include "core/linalg.h"
#include "core/base.h"
#include "core/oracle.h"
#include "core/set_utils.h"
#include "core/sfm_algorithm.h"

namespace submodular {

template <typename ValueType>
class FWRobust: public SFMAlgorithm<ValueType> {
public:
  using value_type = typename ValueTraits<ValueType>::value_type;
  using rational_type = typename ValueTraits<ValueType>::rational_type;
  //using base_type = typename ValueTraits<ValueType>::base_type;
  using base_type = std::vector<rational_type>;

  FWRobust(): precision_(0.5), tol_(1e-10), x_data_(0) {}
  explicit FWRobust(rational_type precision): precision_(precision), tol_(1e-10), x_data_(0) {}

  void Minimize(SubmodularOracle<ValueType>& F);

  std::string GetName() { return "Fujishige--Wolfe"; }

  void SetTol(rational_type tol) { tol_ = tol; }

private:
  rational_type precision_; // = 2 * n_ * epsilon_
  rational_type eps_;
  rational_type tol_;

  base_type x_data_; // x
  std::vector<base_type> bases_;
  std::vector<rational_type> coeffs_;
  std::vector<rational_type> y_;
  std::vector<rational_type> alpha_;
  std::size_t n_;
  std::size_t n_ground_;
  Set domain_;
  std::vector<element_type> members_;
  std::vector<std::size_t> inverse_;

  void Initialize(SubmodularOracle<ValueType>& F);
  bool CheckNorm(const base_type& q);
  bool IsConvexCombination(const std::vector<rational_type>& alpha);
  void CalcAffineMinimizer();
  void FWUpdate(const base_type& q);
  Set GetX() const;
};


template <typename ValueType>
void FWRobust<ValueType>::Initialize(SubmodularOracle<ValueType>& F) {
  domain_ = std::move(F.GetDomain());
  members_ = std::move(domain_.GetMembers());
  inverse_ = std::move(domain_.GetInverseMap());
  n_ = F.GetN();
  n_ground_ = F.GetNGround();
  bases_.clear();
  bases_.reserve(n_);
  coeffs_.clear();
  coeffs_.reserve(n_);
  eps_ = precision_ / static_cast<rational_type>(2 * n_);
}

template <typename ValueType>
bool FWRobust<ValueType>::CheckNorm(const base_type& q) {
  rational_type diff(0); // <x, x - q>
  for (const auto& i: members_) {
    diff += x_data_[inverse_[i]] * (x_data_[inverse_[i]] - q[inverse_[i]]);
  }
  return diff <= eps_ * eps_;
}

template <typename ValueType>
bool FWRobust<ValueType>::IsConvexCombination(const std::vector<rational_type>& alpha) {
  // NOTE: Despite the name, this method actually checks if coeffs_ >= 0
  for (std::size_t i = 0; i < alpha.size(); ++i) {
    if (alpha[i] < 0) {
      return false;
    }
  }
  return true;
}

template <typename ValueType>
void FWRobust<ValueType>::CalcAffineMinimizer() {
  linalg::calc_affine_minimizer(n_, bases_, alpha_, y_); // alpha_ and y_ are changed
}

template <typename ValueType>
void FWRobust<ValueType>::FWUpdate(const base_type& q) {
  bases_.push_back(q);
  coeffs_.emplace_back(0);

  while (true) {// minor cycle
    CalcAffineMinimizer(); // update alpha_ and y_
    auto m = bases_.size();

    if (IsConvexCombination(alpha_)) {
      break;
    }

    rational_type theta = std::numeric_limits<rational_type>::max();
    for (std::size_t i = 0; i < m; ++i) {
      if (alpha_[i] < 0) {
        theta = std::min(theta, coeffs_[i] / (coeffs_[i] - alpha_[i]));
      }
    }
    for (std::size_t i = 0; i < m; ++i) {
      coeffs_[i] = theta * alpha_[i] + (1.0 - theta) * coeffs_[i];
    }

    // remove bases that are no longer used
    for (std::size_t i = m; i != 0; --i) {
      if (utils::is_abs_close(coeffs_[i - 1], 0.0, tol_)) {
        if (i - 1 != m - 1) {
          coeffs_[i - 1] = coeffs_[m - 1];
          bases_[i - 1] = std::move(bases_[m - 1]);
        }
        coeffs_.pop_back();
        bases_.pop_back();
        m--;
      }
    }
  }//minor cycle

  x_data_ = y_;
}

template <typename ValueType>
void FWRobust<ValueType>::Minimize(SubmodularOracle<ValueType>& F) {
  this->reporter_.SetNames(GetName(), F.GetName());
  this->reporter_.EntryTimer(ReportKind::TOTAL);
  this->reporter_.EntryTimer(ReportKind::ORACLE);
  this->reporter_.EntryCounter(ReportKind::ORACLE);
  this->reporter_.EntryTimer(ReportKind::BASE);
  this->reporter_.EntryCounter(ReportKind::BASE);
  //this->reporter_.EntryCounter(ReportKind::ITERATION);

  this->reporter_.TimerStart(ReportKind::TOTAL);

  Initialize(F);

  auto order = LinearOrder(domain_);
  x_data_ = std::move(GreedyBaseData(F, order, inverse_, &(this->reporter_)));
  bases_.push_back(x_data_);
  coeffs_.emplace_back(1);

  base_type vertex_new = std::move(LinearMinimizerData(F, x_data_, members_, inverse_, &(this->reporter_)));

  while (!CheckNorm(vertex_new)) {
    FWUpdate(vertex_new);
    vertex_new = std::move(LinearMinimizerData(F, x_data_, members_, inverse_, &(this->reporter_)));
  }

  auto X = GetX();
  auto minimum_value = F.Call(X, &(this->reporter_));

  this->reporter_.TimerStop(ReportKind::TOTAL);
  this->SetResults(minimum_value, X);
}

template <typename ValueType>
Set FWRobust<ValueType>::GetX() const {
  auto order = GetAscendingOrder(x_data_, members_, inverse_);
  Set X = Set::MakeEmpty(n_ground_);
  for (std::size_t i = 0; i < order.size() - 1; ++i) {
    X.AddElement(order[i]);
    if (x_data_[inverse_[order[i + 1]]] >= 0
        && static_cast<rational_type>(n_ * (x_data_[inverse_[order[i + 1]]] - x_data_[inverse_[order[i]]])) >= eps_) {
      break;
    }
  }
  return X;
}

}

#endif
