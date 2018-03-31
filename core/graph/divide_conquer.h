#ifndef DIVIDE_CONQUER_H
#define DIVIDE_CONQUER_H

#include <vector>
#include <tuple>
#include <utility>
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

#include "core/utils.h"
#include "core/oracle.h"
#include "core/set_utils.h"
#include "core/partial_vector.h"
#include "core/graph/generalized_cut.h"

namespace submodular {

template <class ValueType>
class DivideConquerMNP {
public:
  using value_type = typename ValueTraits<ValueType>::value_type;
  using rational_type = typename ValueTraits<ValueType>::rational_type;
  using base_type = typename ValueTraits<ValueType>::base_type;

  DivideConquerMNP(): tol_(1e-8) {}

  void SetTol(rational_type tol) { tol_ = tol; }

  base_type Solve(GeneralizedCutOracle<ValueType>& F);

private:
  rational_type tol_;
  void Slice(GeneralizedCutOracle<ValueType>& F, std::vector<rational_type>& x);
};

template <typename ValueType>
typename DivideConquerMNP<ValueType>::base_type
DivideConquerMNP<ValueType>::Solve(GeneralizedCutOracle<ValueType>& F) {
  auto n = F.GetN();
  auto domain = F.GetDomain();
  std::vector<rational_type> x(n, 0);
  Slice(F, x);
  base_type base(domain);
  base.SetActiveVector(x);
  return base;
}

template <typename ValueType>
void DivideConquerMNP<ValueType>::Slice(GeneralizedCutOracle<ValueType>& F, std::vector<rational_type>& x) {
  auto indices = F.GetVariableIndices();

  if (indices.size() == 1) {
    auto i = indices[0];
    x[i] = static_cast<rational_type>(F.FV());
  }
  else if (indices.size() > 1) {

    auto n = static_cast<rational_type>(indices.size());
    rational_type alpha = static_cast<rational_type>(F.FV()) / n;

    F.AddCardinalityFunction(-alpha);
    auto fc_minimum_value = static_cast<rational_type>(F.GetMinimumValue());
    auto fc_minimizer_ids = F.GetMinimizerIds();
    F.AddCardinalityFunction(alpha);

    if (utils::is_abs_close(fc_minimum_value, 0.0, tol_)) {
      for (const auto& i: indices) {
        x[i] = alpha;
      }
    }
    else {
      if (fc_minimizer_ids.size() > 0) {
        auto state = F.GetState();
        F.ReductionByIds(fc_minimizer_ids);
        Slice(F, x);
        F.Restore(state);
      }

      if (indices.size() - fc_minimizer_ids.size() > 0) {
        auto state = F.GetState();
        auto F_A = fc_minimum_value + static_cast<rational_type>(alpha * fc_minimizer_ids.size());
        F.ContractionByIds(fc_minimizer_ids, -F_A);
        Slice(F, x);
        F.Restore(state);
      }
    }
  }
}

}

#endif
