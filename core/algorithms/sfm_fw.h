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
  using base_type = typename ValueTraits<ValueType>::base_type;

  FWRobust(): precision_(0.5), tol_(1e-10), x_(0) {}
  explicit FWRobust(rational_type precision): precision_(precision), tol_(1e-10), x_(0) {}

  void Minimize(SubmodularOracle<ValueType>& F);

  void SetTol(rational_type tol) { tol_ = tol; }

private:
  rational_type precision_; // = 2 * n_ * epsilon_
  rational_type eps_;
  rational_type tol_;

  base_type x_; // x
  //base_type q_;
  //BaseCombination<ValueType> combination_;
  std::vector<base_type> bases_;
  std::vector<rational_type> coeffs_;
  std::vector<rational_type> y_;
  std::vector<rational_type> alpha_;
  std::size_t n_;
  std::size_t n_ground_;
  Set domain_;

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
  auto members = domain_.GetMembers();
  rational_type diff(0); // <x, x - q>
  for (const auto& i: members) {
    diff += x_[i] * (x_[i] - q[i]);
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
  std::vector<std::vector<double>> Y;
  Y.reserve(bases_.size());
  for (std::size_t i = 0; i < bases_.size(); ++i) {
    auto data = bases_[i].GetActiveVector();
    //std::vector<double> data_d(data.begin(), data.end());
    Y.push_back(std::move(data));
  }
  linalg::calc_affine_minimizer(n_, Y, alpha_, y_);
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

  x_.SetActiveVector(y_);
}

template <typename ValueType>
void FWRobust<ValueType>::Minimize(SubmodularOracle<ValueType>& F) {
  Initialize(F);

  auto order = LinearOrder(domain_);
  x_ = std::move(GreedyBase(F, order));
  bases_.push_back(x_);
  coeffs_.emplace_back(1);

  base_type vertex_new = std::move(LinearMinimizer(F, x_)); // ++base call

  int work_count = 0;
  while (!CheckNorm(vertex_new)) {
    //std::cout << "work_count = " << work_count << std::endl;
    FWUpdate(vertex_new);
    vertex_new = std::move(LinearMinimizer(F, x_)); // ++base call
    //work_count++;
  }

  auto X = GetX();
  auto minimum_value = F.Call(X);
  this->SetResults(minimum_value, X);
}

template <typename ValueType>
Set FWRobust<ValueType>::GetX() const {
  auto order = GetAscendingOrder(x_);
  Set X = Set::MakeEmpty(n_ground_);
  for (std::size_t i = 0; i < order.size() - 1; ++i) {
    X.AddElement(order[i]);
    if (x_[order[i + 1]] >= 0 && static_cast<rational_type>(n_ * (x_[order[i + 1]] - x_[order[i]])) >= eps_) {
      break;
    }
  }
  return X;
}

}

#endif
