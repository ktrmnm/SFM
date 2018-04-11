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

#ifndef HYPERGRAPH_CUT_H
#define HYPERGRAPH_CUT_H

#include <vector>
#include <set>
#include <stdexcept>
#include <algorithm>
#include <numeric>

#include "core/utils.h"
#include "core/set_utils.h"
#include "core/graph/maxflow.h"
#include "core/graph/generalized_cut.h"

namespace submodular {

template <typename ValueType> class HypergraphCutPlusModular;


template <typename ValueType>
class HypergraphCutPlusModular: public GeneralizedCutOracle<ValueType> {
public:
  HypergraphCutPlusModular(): GeneralizedCutOracle<ValueType>() {}

  std::string GetName() { return "Hypergraph cut (represented as a generalized cut)"; }

  static HypergraphCutPlusModular<ValueType> FromHyperEdgeList(
    std::size_t n,
    const std::vector<std::set<element_type>>& hyperedges,
    const std::vector<ValueType>& capacities,
    const std::vector<ValueType>& modular_term
  );

};


template <typename ValueType>
HypergraphCutPlusModular<ValueType>
HypergraphCutPlusModular<ValueType>::FromHyperEdgeList(
  std::size_t n,
  const std::vector<std::set<element_type>>& hyperedges,
  const std::vector<ValueType>& capacities,
  const std::vector<ValueType>& modular_term
)
{
  auto m = hyperedges.size();
  if (m != capacities.size()) {
    throw std::invalid_argument(
      "HypergraphCutPlusModular::FromHyperEdgeList: hyperedges.size() and capacities.size() do not match"
    );
  }
  if (modular_term.size() != n) {
    throw std::invalid_argument("HypergraphCutPlusModular::FromHyperEdgeList: modular_term.size() should be n");
  }
  auto modular_max = utils::abs_max(modular_term);
  auto cap_max = utils::abs_max(capacities);
  auto edge_num_bound = 6 * (n + 1) * (m + 1);
  auto inf = static_cast<ValueType>(edge_num_bound * std::max(modular_max, cap_max));

  MaxflowGraph<ValueType> graph;
  graph.Reserve(n + 2 * m + 2, edge_num_bound);

  // NOTE: In order to represent a hypergraph cut by a generalized cut function,
  // we add the following (2m + 2) auxiliary nodes:
  // - W1 (id = n, ... n + m - 1)
  // - W2 (id = n + m, ... n + 2m - 1)
  // - source (id = n + 2m)
  // - sink (id = n + 2m - 1)
  std::vector<std::size_t> w1_id(m), w2_id(m);
  std::iota(w1_id.begin(), w1_id.end(), n);
  std::iota(w2_id.begin(), w2_id.end(), n + m);
  std::size_t s = n + 2 * m;
  std::size_t t = n + 2 * m + 1;
  for (element_type i = 0; i < n; ++i) {
    graph.AddNode(i, true); // Add n "variable" nodes
  }
  for (std::size_t i = n; i <= t; ++i) {
    graph.AddNode(i, false); // Add "non-variable" nodes
  }
  auto source = graph.GetNodeById(s);
  auto sink = graph.GetNodeById(t);

  // Make modular part
  for (std::size_t node_id = 0; node_id < n; ++node_id) {
    // NOTE: We don't have to worry about s-v or v-t capacities being non-positive, because they are
    // corrected by the maxflow algorithm.
    auto node = graph.GetNodeById(node_id);
    auto xi = modular_term[node_id];
    graph.AddVTArcPair(sink, node, xi, 0);
    graph.AddSVArcPair(node, source, 0, 0);
  }

  // Make hypergraph part
  for (std::size_t edge_id = 0; edge_id < m; ++edge_id) {
    auto w1 = graph.GetNodeById(w1_id[edge_id]);
    auto w2 = graph.GetNodeById(w2_id[edge_id]);
    auto cap = capacities[edge_id];
    if (cap < 0) {
      throw std::invalid_argument("HypergraphCutPlusModular::FromHyperEdgeList: Negative inner capacity");
    }

    // add w1 -> sink
    graph.AddVTArcPair(sink, w1, cap, 0);
    // add source -> w2
    graph.AddSVArcPair(w2, source, cap, 0);

    for (const auto& i: hyperedges[edge_id]) {
      if (i >= n) {
        throw std::invalid_argument("HypergraphCutPlusModular::FromHyperEdgeList: Node names should be less than n");
      }
      auto node = graph.GetNodeById(i);
      // add node -> w1
      graph.AddArcPair(w1, node, cap, 0);
      // add w2 -> node with infinite capacity
      graph.AddArcPair(node, w2, inf, 0);
    }
  }
  graph.MakeGraph(source, sink);
  HypergraphCutPlusModular<ValueType> F;
  F.SetGraph(std::move(graph));
  return F;
}

}

#endif
