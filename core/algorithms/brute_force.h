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
  value_type min_value = std::numeric_limits<value_type>::max();
  auto n_ground = F.GetNGround();
  Set minimizer(n_ground);

  this->ClearStats();
  auto start_time = SimpleTimer::Now();

  for (unsigned long i = 0; i < (1 << n_ground); ++i) {
    Set X(n_ground, i);

    auto oracle_start = SimpleTimer::Now();
    auto new_value = F.Call(X);
    auto oracle_end = SimpleTimer::Now();

    this->IncreaseOracleTime(oracle_start - oracle_end);
    this->IncreaseOracleCount(1);
    if (new_value < min_value) {
      min_value = new_value;
      minimizer = X;
    }
  }

  auto end_time = SimpleTimer::Now();

  this->IncreaseTotalTime(start_time - end_time);
  this->SetResults(min_value, minimizer);
}

}

#endif
