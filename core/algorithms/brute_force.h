#ifndef ALGORITHMS_BRUTE_FORCE_H
#define ALGORITHMS_BRUTE_FORCE_H

#include <limits>
#include "core/oracle.h"
#include "core/set_utils.h"
#include "core/sfm_algorithm.h"

namespace submodular {

template <typename ValueType>
class BruteForce: public SFMAlgorithm<ValueType> {
public:
  using value_type = typename ValueTraits<ValueType>::value_type;

  BruteForce() = default;

  void Minimize(SubmodularOracle<ValueType>& F);

};

template <typename ValueType>
void BruteForce<ValueType>::Minimize(SubmodularOracle<ValueType>& F) {
  this->reporter_.EntryTimer(ReportKind::TOTAL);
  this->reporter_.EntryTimer(ReportKind::ORACLE);
  this->reporter_.EntryCounter(ReportKind::ORACLE);

  this->reporter_.TimerStart(ReportKind::TOTAL);

  value_type min_value = std::numeric_limits<value_type>::max();
  auto n_ground = F.GetNGround();
  Set minimizer(n_ground);

  for (unsigned long i = 0; i < (1 << n_ground); ++i) {
    Set X(n_ground, i);

    auto new_value = F.Call(X, &(this->reporter_));

    if (new_value < min_value) {
      min_value = new_value;
      minimizer = X;
    }
  }

  this->reporter_.TimerStop(ReportKind::TOTAL);
  this->SetResults(min_value, minimizer);
}

}

#endif
