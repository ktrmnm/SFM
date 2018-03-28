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

  // Get node indices in the internal graph object such that is_variable == true.
  // Do not confuse with GetMembers()
  std::vector<std::size_t> GetVariableIndices() const;
  // Get domain members.
  std::vector<element_type> GetMembers() const;

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
  std::size_t node_number_;
  MaxflowGraph<ValueType> graph_;
  bool done_minimize_;
  value_type minimum_value_;
  std::vector<std::size_t> minimizer_ids_;
  //Set minimizer_;
};

template <typename ValueType>
void GeneralizedCutOracle<ValueType>::SetGraph(const MaxflowGraph<ValueType>& graph) {
  graph_ = graph;
  node_number_ = graph_.GetNodeNumber();
}

template <typename ValueType>
void GeneralizedCutOracle<ValueType>::SetGraph(MaxflowGraph<ValueType>&& graph) {
  graph_ = std::move(graph);
  node_number_ = graph_.GetNodeNumber();
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
GeneralizedCutOracle<ValueType>::value_type
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
std::vector<std::size_t> GeneralizedCutOracle<ValueType>::GetVariableIndices() const {
  return graph_.GetInnerIndices(true);
}

template <typename ValueType>
std::vector<element_type> GeneralizedCutOracle<ValueType>::GetMembers() const {
  return graph_.GetMembers();
}

template <typename ValueType>
GeneralizedCutOracle<ValueType>::value_type
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
GeneralizedCutOracle<ValueType>::value_type
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
void GeneralizedCutOracle<ValueType>::ReductionByIds(const std::vector<std::size_t> &A) {
  graph_.ReductionByIds(A);
  done_minimize_ = false;
}

template <typename ValueType>
void GeneralizedCutOracle<ValueType>::ContractionByIds(const std::vector<std::size_t> &A, ValueType offset) {
  graph_.ContractionByIds(A, offset);
  done_minimize_ = false;
}

template <typename ValueType>
GeneralizedCutOracle<ValueType>::state_type
GeneralizedCutOracle<ValueType>::GetState() const {
  return graph_.GetState();
}

template <typename ValueType>
void GeneralizedCutOracle<ValueType>::Restore(GraphState<ValueType> state) {
  graph_.RestoreState(state);
  done_minimize_ = false;
}


template <typename ValueType>
class SFMAlgorithmGeneralizedCut {
public:
  using value_type = typename ValueTraits<ValueType>::value_type;

  SFMAlgorithm(): done_sfm_(false) {}

  void Minimize(GeneralizedCutOracle<ValueType>& F);

  value_type GetMinimumValue();
  Set GetMinimizer();
  //SFMReporter GetReporter();

protected:
  bool done_sfm_;
  //SFMReporter reporter_;
  value_type minimum_value_;
  Set minimizer_;
  //void ClearReports();
  //void SetResults(value_type minimum_value, const Set& minimizer);
};

template <typename ValueType>
void SFMAlgorithmGeneralizedCut<ValueType>::Minimize(GeneralizedCutOracle<ValueType>& F) {
  if (!done_sfm_) {
    minimum_value_ = F.GetMinimumValue();
    auto minimizer_members = F.GetMinimizerMembers();
    auto n_ground = F.GetNGround();
    minimizer_ = Set::FromIndices(n_ground, minimizer_members);
    done_sfm_ = true;
  }
}

template <typename ValueType>
SFMAlgorithmGeneralizedCut<ValueType>::value_type
SFMAlgorithmGeneralizedCut<ValueType>::GetMinimumValue() {
  return minimum_value_;
}

template <typename ValueType>
Set SFMAlgorithmGeneralizedCut<ValueType>::GetMinimizer() {
  return minimizer_;
}

}

#endif
