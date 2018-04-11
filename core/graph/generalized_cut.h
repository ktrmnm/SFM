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

#ifndef GENERALIZED_CUT_H
#define GENERALIZED_CUT_H

#include <vector>
#include <tuple>
#include <utility>
#include <algorithm>
#include <numeric>

#include "core/utils.h"
#include "core/set_utils.h"
#include "core/oracle.h"
//#include "core/sfm_algorithm.h"
#include "core/graph/maxflow.h"
//#include "core/reporter.h"

namespace submodular {

template <typename ValueType> class GeneralizedCutOracle;
template <typename ValueType> class SFMAlgorithmGeneralizedCut;

template <typename ValueType>
class GeneralizedCutOracle: public SubmodularOracle<ValueType> {
public:
  using value_type = typename ValueTraits<ValueType>::value_type;
  using rational_type = typename ValueTraits<ValueType>::rational_type;
  using state_type = MaxflowState<ValueType>;

  GeneralizedCutOracle(): done_minimize_(false) {}
  GeneralizedCutOracle(const GeneralizedCutOracle&) = default;
  GeneralizedCutOracle(GeneralizedCutOracle&&) = default;
  GeneralizedCutOracle& operator=(const GeneralizedCutOracle&) = default;
  GeneralizedCutOracle& operator=(GeneralizedCutOracle&&) = default;

  void SetGraph(const MaxflowGraph<ValueType>& graph);
  void SetGraph(MaxflowGraph<ValueType>&& graph);
  void AddCardinalityFunction(value_type multiplier);

  std::size_t GetN() const;
  std::size_t GetNGround() const;
  value_type Call(const Set& X);
  virtual std::string GetName() { return "Generalized Cut"; }

  // Get node indices in the internal graph object such that is_variable == true.
  // Do not confuse with GetMembers()
  std::vector<std::size_t> GetVariableIndices() const;
  // Get domain members.
  std::vector<element_type> GetMembers();

  value_type FV();
  void Minimize();
  value_type GetMinimumValue();
  std::vector<std::size_t> GetMinimizerIds();
  std::vector<element_type> GetMinimizerMembers();
  void ReductionByIds(const std::vector<std::size_t>& A);
  void ContractionByIds(const std::vector<std::size_t>& A, value_type offset = 0);

  state_type GetState() const;
  void Restore(state_type state);

protected:
  //std::size_t node_number_;
  std::size_t n_ground_;
  MaxflowGraph<ValueType> graph_;
  bool done_minimize_;
  value_type minimum_value_;
  std::vector<std::size_t> minimizer_ids_;
  //Set minimizer_;
  void MakeDomain();
};

template <typename ValueType>
void GeneralizedCutOracle<ValueType>::MakeDomain() {
  Set domain = Set::MakeEmpty(n_ground_);
  for (auto&& node_id: GetVariableIndices()) {
    domain.AddElement(graph_.Id2Name(node_id));
  }
  this->SetDomain(std::move(domain));
}

template <typename ValueType>
void GeneralizedCutOracle<ValueType>::SetGraph(const MaxflowGraph<ValueType>& graph) {
  graph_ = graph;
  n_ground_ = graph_.GetNGround();
  MakeDomain();
}

template <typename ValueType>
void GeneralizedCutOracle<ValueType>::SetGraph(MaxflowGraph<ValueType>&& graph) {
  graph_ = std::move(graph);
  n_ground_ = graph_.GetNGround();
  MakeDomain();
}

template <typename ValueType>
void GeneralizedCutOracle<ValueType>::AddCardinalityFunction(value_type multiplier) {
  graph_.AddCardinalityFunction(multiplier);
}

template <typename ValueType>
std::size_t GeneralizedCutOracle<ValueType>::GetN() const {
  return GetVariableIndices().size();
}

template <typename ValueType>
std::size_t GeneralizedCutOracle<ValueType>::GetNGround() const {
  return graph_.GetNGround();
}

template <typename ValueType>
typename GeneralizedCutOracle<ValueType>::value_type
GeneralizedCutOracle<ValueType>::Call(const Set& X) {
  auto members = X.GetMembers();
  value_type val(0);
  if (!graph_.HasAuxiliaryNodes()) {
    val = graph_.GetCutValueByNames(members);
  }
  else {
    auto state1 = graph_.GetState();
    graph_.ReductionByNames(members);
    auto state2 = graph_.GetState();
    graph_.ContractionByNames(members, value_type(0));
    val = graph_.GetMaxFlowValue();
    graph_.RestoreState(state2);
    graph_.RestoreState(state2);
  }
  return val;
}

template <typename ValueType>
std::vector<std::size_t> GeneralizedCutOracle<ValueType>::GetVariableIndices() const{
  return graph_.GetInnerIndices(true);
}

template <typename ValueType>
std::vector<element_type> GeneralizedCutOracle<ValueType>::GetMembers() {
  return graph_.GetMembers();
}

template <typename ValueType>
typename GeneralizedCutOracle<ValueType>::value_type
GeneralizedCutOracle<ValueType>::FV() {
  auto V = GetVariableIndices();
  auto state = graph_.GetState();
  graph_.ContractionByIds(V);
  auto value = graph_.GetMaxFlowValue();
  graph_.RestoreState(state);
  return value;
}

template <typename ValueType>
void GeneralizedCutOracle<ValueType>::Minimize() {
  if (!done_minimize_) {
    minimum_value_ = graph_.GetMaxFlowValue();
    minimizer_ids_ = graph_.GetMinCut(true);
    done_minimize_ = true;
  }
}

template <typename ValueType>
typename GeneralizedCutOracle<ValueType>::value_type
GeneralizedCutOracle<ValueType>::GetMinimumValue() {
  if (!done_minimize_) {
    Minimize();
  }
  return minimum_value_;
}

template <typename ValueType>
std::vector<std::size_t> GeneralizedCutOracle<ValueType>::GetMinimizerIds() {
  if (!done_minimize_) {
    Minimize();
  }
  return minimizer_ids_;
}

template <typename ValueType>
std::vector<element_type> GeneralizedCutOracle<ValueType>::GetMinimizerMembers() {
  if (!done_minimize_) {
    Minimize();
  }
  std::vector<element_type> members;
  for (const auto& node_id: minimizer_ids_) {
    members.push_back(graph_.Id2Name(node_id));
  }
  return members;
}

template <typename ValueType>
void GeneralizedCutOracle<ValueType>::ReductionByIds(const std::vector<std::size_t>& A) {
  graph_.ReductionByIds(A);
  done_minimize_ = false;
}

template <typename ValueType>
void GeneralizedCutOracle<ValueType>::ContractionByIds(const std::vector<std::size_t>& A, value_type offset) {
  graph_.ContractionByIds(A, offset);
  done_minimize_ = false;
}

template <typename ValueType>
typename GeneralizedCutOracle<ValueType>::state_type
GeneralizedCutOracle<ValueType>::GetState() const {
  return graph_.GetState();
}

template <typename ValueType>
void GeneralizedCutOracle<ValueType>::Restore(state_type state) {
  graph_.RestoreState(state);
  done_minimize_ = false;
}


template <typename ValueType>
class SFMAlgorithmGeneralizedCut {
public:
  using value_type = typename ValueTraits<ValueType>::value_type;

  SFMAlgorithmGeneralizedCut(): done_sfm_(false) {}
  void SetReporter(const SFMReporter& reporter) { reporter_ = reporter; }
  void SetReporter(SFMReporter&& reporter) { reporter_ = std::move(reporter); }

  void Minimize(GeneralizedCutOracle<ValueType>& F);

  std::string GetName() { return "Maxflow"; }

  value_type GetMinimumValue();
  Set GetMinimizer();
  SFMReporter GetReporter() { return reporter_; }

protected:
  bool done_sfm_;
  SFMReporter reporter_;
  //value_type minimum_value_;
  //Set minimizer_;
  //void ClearReports();
  //void SetResults(value_type minimum_value, const Set& minimizer);
};

template <typename ValueType>
void SFMAlgorithmGeneralizedCut<ValueType>::Minimize(GeneralizedCutOracle<ValueType>& F) {
  if (!done_sfm_) {
    this->reporter_.SetNames(GetName(), F.GetName());
    this->reporter_.EntryTimer(ReportKind::TOTAL);
    this->reporter_.TimerStart(ReportKind::TOTAL);

    auto minimum_value = static_cast<double>(F.GetMinimumValue());
    auto minimizer_members = F.GetMinimizerMembers();
    auto n_ground = F.GetNGround();
    auto X = Set::FromIndices(n_ground, minimizer_members);
    done_sfm_ = true;

    this->reporter_.TimerStop(ReportKind::TOTAL);
    this->reporter_.minimum_value_ = minimum_value;
    this->reporter_.minimizer_ = std::move(X);
  }
}

template <typename ValueType>
typename SFMAlgorithmGeneralizedCut<ValueType>::value_type
SFMAlgorithmGeneralizedCut<ValueType>::GetMinimumValue() {
  return static_cast<value_type>(reporter_.minimum_value_);
}

template <typename ValueType>
Set SFMAlgorithmGeneralizedCut<ValueType>::GetMinimizer() {
  return reporter_.minimizer_;
}

}

#endif
