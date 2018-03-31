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

#ifndef SFM_SOLVER_H
#define SFM_SOLVER_H

#include <chrono>
#include <utility>
#include <string>
#include "core/set_utils.h"
#include "core/oracle.h"
#include "core/reporter.h"

namespace submodular {

template <typename ValueType> class SFMAlgorithm;
template <typename ValueType> class SFMAlgorithmWithReduction;

// Base class of SFM algorithms
template <typename ValueType>
class SFMAlgorithm {
public:
  using value_type = typename ValueTraits<ValueType>::value_type;

  SFMAlgorithm(): done_sfm_(false) {}
  //explicit SFMAlgorithm(const SFMReporter& reporter): reporter_(reporter), done_sfm_(false) {}
  //explicit SFMAlgorithm(SFMReporter&& reporter): reporter(std::move(reporter)), done_sfm(false) {}
  void SetReporter(const SFMReporter& reporter) { reporter_ = reporter; }
  void SetReporter(SFMReporter&& reporter) { reporter_ = std::move(reporter); }

  // Perform SFM algorithm.
  // The minimum value (and a minimizer) should be stored in minimum_value (resp. minimizer_)
  // Some statistics should be reported as stats_.
  // NOTE: SubmodularOracle object F will possibly be modified by the algorithm.
  virtual void Minimize(SubmodularOracle<ValueType>& F) = 0;

  virtual std::string GetName() = 0;

  value_type GetMinimumValue();
  Set GetMinimizer();
  SFMReporter GetReporter();

protected:
  bool done_sfm_;
  SFMReporter reporter_;
  //value_type minimum_value_;
  //Set minimizer_;
  void ClearReports();
  void SetResults(value_type minimum_value, const Set& minimizer);
};

template <typename ValueType>
typename SFMAlgorithm<ValueType>::value_type
SFMAlgorithm<ValueType>::GetMinimumValue() {
  return static_cast<value_type>(reporter_.minimum_value_);
}

template <typename ValueType>
Set SFMAlgorithm<ValueType>::GetMinimizer() {
  return reporter_.minimizer_;
}

template <typename ValueType>
SFMReporter SFMAlgorithm<ValueType>::GetReporter() {
  return reporter_;
}

template <typename ValueType>
void SFMAlgorithm<ValueType>::ClearReports() {
  reporter_.Clear();
}

template <typename ValueType>
void SFMAlgorithm<ValueType>::SetResults(value_type minimum_value, const Set& minimizer) {
    //minimum_value_ = minimum_value;
    //minimizer_ = minimizer;
    reporter_.SetResults(static_cast<double>(minimum_value), minimizer);
    done_sfm_ = true;
}

template <typename ValueType>
class SFMAlgorithmWithReduction {
public:
  using value_type = typename ValueTraits<ValueType>::value_type;

  SFMAlgorithmWithReduction(): done_sfm_(false) {}

  virtual void Minimize(ReducibleOracle<ValueType>& F) = 0;

  virtual std::string GetName() = 0;

  value_type GetMinimumValue();
  Set GetMinimizer();
  SFMReporter GetReporter();

protected:
  bool done_sfm_;
  //value_type minimum_value_;
  SFMReporter reporter_;
  //Set minimizer_;
  void ClearReports();
  void SetResults(value_type minimum_value, const Set& minimizer);
};

template <typename ValueType>
typename SFMAlgorithmWithReduction<ValueType>::value_type
SFMAlgorithmWithReduction<ValueType>::GetMinimumValue() {
  return static_cast<value_type>(reporter_.minimum_value_);
}

template <typename ValueType>
Set SFMAlgorithmWithReduction<ValueType>::GetMinimizer() {
  return reporter_.minimizer_;
}

template <typename ValueType>
void SFMAlgorithmWithReduction<ValueType>::SetResults(value_type minimum_value, const Set& minimizer) {
    //minimum_value_ = minimum_value;
    //minimizer_ = minimizer;
    reporter_.SetResults(static_cast<double>(minimum_value), minimizer);
    done_sfm_ = true;
}

template <typename ValueType>
SFMReporter SFMAlgorithmWithReduction<ValueType>::GetReporter() {
  return reporter_;
}

template <typename ValueType>
void SFMAlgorithmWithReduction<ValueType>::ClearReports() {
  reporter_.Clear();
}

}

#endif
