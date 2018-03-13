#ifndef SFM_SOLVER_H
#define SFM_SOLVER_H

#include <chrono>
#include <utility>
#include "core/set_utils.h"
#include "core/oracle.h"

namespace submodular {

struct SimpleTimer {
  static auto Now() { return std::chrono::system_clock::now(); }
};

struct SFMStats {
  unsigned int oracle_calls; // number of oracle calls
  unsigned int base_calls; // number of base calls
  std::chrono::milliseconds oracle_time; // time spent by oracle calls
  std::chrono::milliseconds base_time; // time spent by base calls
  std::chrono::milliseconds total_time; // time spent by overall calculation
};

// Base class of SFM algorithms
template <typename ValueType>
class SFMAlgorithm {
public:
  using value_type = typename ValueTraits<ValueType>::value_type;

  SFMAlgorithm(): done_sfm_(false) {}

  // Perform SFM algorithm.
  // The minimum value (and a minimizer) should be stored in minimum_value (resp. minimizer_)
  // Some statistics should be reported as stats_.
  // NOTE: SubmodularOracle object F will possibly be modified by the algorithm.
  virtual void Minimize(SubmodularOracle<ValueType>& F) = 0;

  value_type GetMinimumValue();
  Set GetMinimizer();
  SFMStats GetStats();

protected:
  bool done_sfm_;
  SFMStats stats_;
  value_type minimum_value_;
  Set minimizer_;
  void ClearStats();
  void IncreaseOracleCount(unsigned int count);
  void IncreaseBaseCount(unsigned int count);
  void IncreaseOracleTime(std::chrono::duration<double> time_delta);
  void IncreaseBaseTime(std::chrono::duration<double> time_delta);
  void IncreaseTotalTime(std::chrono::duration<double> time_delta);
  void SetResults(value_type minimum_value, const Set& minimizer);
};

template <typename ValueType>
typename SFMAlgorithm<ValueType>::value_type
SFMAlgorithm<ValueType>::GetMinimumValue() {
  return minimum_value_;
}

template <typename ValueType>
Set SFMAlgorithm<ValueType>::GetMinimizer() {
  return minimizer_;
}

template <typename ValueType>
SFMStats SFMAlgorithm<ValueType>::GetStats() {
  return stats_;
}

template <typename ValueType>
void SFMAlgorithm<ValueType>::ClearStats() {
  stats_.oracle_calls = 0;
  stats_.base_calls = 0;
  stats_.oracle_time = std::chrono::milliseconds::zero();
  stats_.base_time = std::chrono::milliseconds::zero();
  stats_.total_time = std::chrono::milliseconds::zero();
}

template <typename ValueType>
void SFMAlgorithm<ValueType>::IncreaseOracleCount(unsigned int count) {
  stats_.oracle_calls += count;
}

template <typename ValueType>
void SFMAlgorithm<ValueType>::IncreaseBaseCount(unsigned int count) {
  stats_.base_calls += count;
}

template <typename ValueType>
void SFMAlgorithm<ValueType>::IncreaseOracleTime(std::chrono::duration<double> time_delta) {
  stats_.oracle_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_delta);
}

template <typename ValueType>
void SFMAlgorithm<ValueType>::IncreaseBaseTime(std::chrono::duration<double> time_delta) {
  stats_.base_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_delta);
}

template <typename ValueType>
void SFMAlgorithm<ValueType>::IncreaseTotalTime(std::chrono::duration<double> time_delta) {
  stats_.total_time += std::chrono::duration_cast<std::chrono::milliseconds>(time_delta);
}

template <typename ValueType>
void SFMAlgorithm<ValueType>::SetResults(value_type minimum_value, const Set& minimizer) {
    minimum_value_ = minimum_value;
    minimizer_ = minimizer;
    done_sfm_ = true;
}

}

#endif
