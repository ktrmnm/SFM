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

  std::string GetName() { return "Brute Force"; }

  void Minimize(SubmodularOracle<ValueType>& F);

};

template <typename ValueType>
void BruteForce<ValueType>::Minimize(SubmodularOracle<ValueType>& F) {
  this->reporter_.SetNames(GetName(), F.GetName());
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
