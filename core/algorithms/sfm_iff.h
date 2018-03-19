#ifndef ALGORITHMS_SFM_IFF_H
#define ALGORITHMS_SFM_IFF_H

#include <utility>
#include <limits>
#include <vector>
#include <deque>
#include <list>
#include <algorithm>
#include <memory>
#include <iostream>
#include "core/base.h"
#include "core/oracle.h"
#include "core/set_utils.h"
#include "core/sfm_algorithm.h"
#include "core/graph.h"

#define DEBUG
#ifdef DEBUG
#include <iostream>
#endif

namespace submodular {

template <typename ValueType> class IFFWP;
//template <typename ValueType> class IFFSP;


template <typename ValueType>
class IFFWP: public SFMAlgorithmWithReduction<ValueType> {
public:
  using value_type = typename ValueTraits<ValueType>::value_type;
  using rational_type = typename ValueTraits<ValueType>::rational_type;
  using base_type = typename ValueTraits<ValueType>::base_type;

  IFFWP(): precision_(1) {};
  explicit IFFWP(rational_type precition): precision_(precition) {}

  void Minimize(ReducibleOracle<ValueType>& F);

protected:
  rational_type precision_; // eta
  rational_type target_delta_;
  rational_type delta_; // current scaling

  //base_type base_; // x
  BaseCombination<ValueType> base_combination_;
  std::size_t active_order_id_;
  std::pair<std::size_t, std::size_t> active_pair_;

  Set S_negative_; // S
  Set S_positive_; // T
  Set S_candidate_; // W

  SimpleGraph<rational_type> G_;
  using Arc_s = typename GraphTraits<SimpleGraph<rational_type>>::Arc_s;
  std::list<Arc_s> augmenting_path_;

  ReducibleOracle<ValueType> F_;
  std::size_t n_;
  std::size_t n_ground_;
  Set domain_;

  void InitializeCommon(ReducibleOracle<ValueType>& F);

  void InitializeWP();

  // Main part of weakly polynomial IFF algorithm
  void Scaling();

  base_type GetAnnealedBase();
  void UpdateST();
  void UpdateW();
  bool FindAugmentingPath();
  void Augment();
  bool FindActiveTriple();
  void DoubleExchange();
  void MultiplyFlow(rational_type multiplier);
};


template <typename ValueType>
void IFFWP<ValueType>::Minimize(ReducibleOracle<ValueType>& F) {
  #ifdef DEBUG
  std::cout << "Minimize start" << std::endl;
  #endif
  //this->ClearStats();
  //auto start_time = SimpleTimer::Now();

  InitializeCommon(F);
  InitializeWP();
  Scaling();
  value_type min_value = F.Call(S_candidate_);
  #ifdef DEBUG
  std::cout << "Minimize end: min_value = " << min_value << std::endl;
  #endif

  //auto end_time = SimpleTimer::Now();
  //this->IncreaseTotalTime(start_time - end_time);
  this->SetResults(min_value, S_candidate_);
}

template <typename ValueType>
void IFFWP<ValueType>::InitializeCommon(ReducibleOracle<ValueType>& F) {
  F_ = F;
  auto domain = F_.GetDomain();
  G_ = std::move(MakeCompleteGraph<rational_type>(domain, 0));
}

template <typename ValueType>
void IFFWP<ValueType>::InitializeWP() {
  domain_ = std::move(F_.GetDomain());
  n_ = F_.GetN();
  n_ground_ = F_.GetNGround();

  base_combination_ = std::move(BaseCombination<ValueType>(domain_));
  auto L = LinearOrder(domain_);
  auto x = GreedyBase(F_, L);

  target_delta_ = precision_ / static_cast<rational_type>(n_ * n_);

  rational_type x_minus = 0;
  rational_type x_plus = 0;
  for (const auto& i: domain_.GetMembers()) {
    if (x[i] >= 0) {
      x_plus += x[i];
    }
    else {
      x_minus -= x[i];
    }
  }
  delta_ = std::min(x_minus, x_plus) / static_cast<rational_type>(n_ * n_);

  auto coeff = rational_type(1);
  base_combination_.AddTriple(std::move(L), std::move(x), coeff);
}

template <typename ValueType>
void IFFWP<ValueType>::Scaling() {
  #ifdef DEBUG
  unsigned int work_count = 0;
  unsigned int max_work_count = 300;
  #endif

  while (delta_ >= target_delta_) {
    #ifdef DEBUG
    if (work_count >= max_work_count) {
      break;
    }
    std::cout << "delta = " << delta_ << " target = " << target_delta_ << std::endl;
    #endif
    UpdateST();

    // Main loop
    while (true) {
      if (FindAugmentingPath()) {
        Augment();
        UpdateST();
      }
      else if (FindActiveTriple()){
        DoubleExchange();
        UpdateST();
      }
      else {
        break;
      }

      //if (base_combination_.Size() >= n_) {
      //  base_combination_.Reduce();
      //}
      base_combination_.Reduce();
      #ifdef DEBUG
      work_count++;
      std::cout << "work count = " << work_count << std::endl;
      if (work_count >= max_work_count) {
        std::cout << "warning: work count exceed " << max_work_count << std::endl;
        break;
      }
      #endif
    }

    delta_ = 0.5 * delta_;
    MultiplyFlow(0.5);
  }

  UpdateW();
}

template <typename ValueType>
typename IFFWP<ValueType>::base_type IFFWP<ValueType>::GetAnnealedBase() {
  #ifdef DEBUG
  std::cout << "GetAnnealedBase" << std::endl;
  #endif
  auto x = base_combination_.GetCombination();
  for (const auto& i: domain_.GetMembers()) {
    x[i] += G_.GetNode(i)->excess;
  }
  #ifdef DEBUG
  std::cout << "x = [";
  for (const auto& i: domain_.GetMembers()) {
    std::cout << x[i] << " ";
  }
  std::cout << "]" << std::endl;
  #endif
  return x;
}

template <typename ValueType>
void IFFWP<ValueType>::UpdateST() {
  #ifdef DEBUG
  std::cout << "UpdateST" << std::endl;
  #endif
  auto y = GetAnnealedBase();
  S_negative_ = std::move(Set::MakeEmpty(n_ground_));
  S_positive_ = std::move(Set::MakeEmpty(n_ground_));
  for (const auto& i: domain_.GetMembers()) {
    if (y[i] >= delta_) {
      S_positive_.AddElement(i);
    }
    if (y[i] <= -delta_) {
      S_negative_.AddElement(i);
    }
  }
  #ifdef DEBUG
  std::cout << "S = ";
  for (const auto& i: S_negative_.GetMembers()) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  std::cout << "T = ";
  for (const auto& i: S_positive_.GetMembers()) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  #endif
}

template <typename ValueType>
void IFFWP<ValueType>::UpdateW() {
  #ifdef DEBUG
  std::cout << "UpdateW" << std::endl;
  #endif
  auto reachable = GetReachableIndicesFrom(G_, S_negative_.GetMembers());
  S_candidate_ = std::move(Set::FromIndices(n_ground_, reachable));
  #ifdef DEBUG
  std::cout << "W = ";
  for (const auto& i: S_candidate_.GetMembers()) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  #endif
}

template <typename ValueType>
bool IFFWP<ValueType>::FindAugmentingPath() {
  #ifdef DEBUG
  std::cout << "FindAugmentingPath" << std::endl;
  #endif
  augmenting_path_ = std::move(FindSTPath(G_, S_negative_, S_positive_));
  return augmenting_path_.size() > 0;
}

template <typename ValueType>
void IFFWP<ValueType>::Augment() {
  #ifdef DEBUG
  std::cout << "Augment" << std::endl;
  #endif
  for (const auto& arc: augmenting_path_) {
    // NOTE: G_.Push preserves skew-symmetry of flows,
    // whereas G_.Augment preserves that flows are non-negative.
    // The former is used in the original IFF paper, and the latter is used in Fujishige (2005) book.
    G_.Augment(arc, delta_);
    //G_.Push(arc, delta_);
  }
}

template <typename ValueType>
bool IFFWP<ValueType>::FindActiveTriple() {
  #ifdef DEBUG
  std::cout << "FindActiveTriple" << std::endl;
  #endif
  UpdateW();
  for (std::size_t order_id = 0; order_id < base_combination_.Size(); ++order_id) {
    if (base_combination_.IsActive(order_id, S_candidate_)) {
      active_order_id_ = order_id;
      active_pair_ = base_combination_.GetActivePair(order_id, S_candidate_);
      #ifdef DEBUG
      std::cout << "active triple found: order = [";
      for (const auto& i: base_combination_.GetOrder(order_id)) {
        std::cout << i << " ";
      }
      std::cout << "], u = " << active_pair_.first << ", v = " << active_pair_.second << std::endl;
      #endif
      return true;
    }
  }
  return false;
}

template <typename ValueType>
void IFFWP<ValueType>::DoubleExchange() {
  std::size_t u = active_pair_.first;
  std::size_t v = active_pair_.second;
  #ifdef DEBUG
  std::cout << "DoubleExchange: u = " << u << " v = " << v << std::endl;
  #endif

  // Calculate exchange capacity
  Set X = Set::MakeEmpty(n_ground_);
  auto order = base_combination_.GetOrder(active_order_id_);
  auto y = base_combination_.GetBase(active_order_id_);
  auto coeff = base_combination_.GetCoeff(active_order_id_);
  std::size_t i_v;
  for (std::size_t i = 0; i < order.size(); ++i) {
    X.AddElement(order[i]);
    if (order[i] == v) {
      i_v = i;
      break;
    }
  }
  rational_type alpha = y[v] + F_.Call(X);
  X.AddElement(u);
  alpha -= F_.Call(X);
  #ifdef DEBUG
  std::cout << "DoubleExchange(1): alpha = " << alpha << std::endl;
  #endif

  // Augment flow
  auto beta = std::min(delta_, coeff * alpha);
  bool is_saturating = coeff * alpha <= delta_;
  auto arc_rev = G_.GetArc(v, u);
  //G_.Push(arc, -beta);
  G_.Augment(arc_rev, beta);
  #ifdef DEBUG
  std::cout << "DoubleExchange(2): beta = " << beta << std::endl;
  #endif

  // Update bases
  if (!is_saturating) {
    base_type y_new(y);
    OrderType order_new(order);
    auto order_id_new = base_combination_.Size();
    auto coeff_new = beta / alpha;
    base_combination_.AddTriple(std::move(order_new), std::move(y_new), coeff_new);

    coeff -= beta / alpha;
    base_combination_.GetCoeff(active_order_id_) = coeff;
  }
  #ifdef DEBUG
  std::cout << "DoubleExchange(4): update" << std::endl;
  #endif
  order[i_v] = u;
  order[i_v + 1] = v;
  base_combination_.GetOrder(active_order_id_) = std::move(order);
  y[v] -= alpha;
  y[u] += alpha;
  base_combination_.GetBase(active_order_id_) = std::move(y);

}

template <typename ValueType>
void IFFWP<ValueType>::MultiplyFlow(rational_type multiplier) {
  for (auto&& node: G_.NodeRange()) {
    for (auto&& arc: G_.OutArcRange(node)) {
      arc->flow *= multiplier;
    }
  }
}

}

#endif
