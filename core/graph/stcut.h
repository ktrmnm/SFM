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

#ifndef STCUT_H
#define STCUT_H

#include <vector>
#include <utility>
#include <stdexcept>

#include "core/set_utils.h"
#include "core/graph/maxflow.h"
#include "core/graph/generalized_cut.h"

namespace submodular {

template <typename ValueType> class STCut;
template <typename ValueType> class STCutPlusModular;
//template <typename ValueType> class Cut;
template <typename ValueType> class CutPlusModular;

enum DirectionKind { DIRECTED = 0, UNDIRECTED = 1 };


template <typename ValueType>
class STCut: public GeneralizedCutOracle<ValueType> {
public:
  STCut(): GeneralizedCutOracle<ValueType>() {}

  std::string GetName() { return "S-T Cut"; }

  static STCut<ValueType> FromEdgeList(
    std::size_t n, element_type s, element_type t,
    const std::vector<std::pair<element_type, element_type>>& edges,
    const std::vector<ValueType>& capacities
  );

};

template <typename ValueType>
class STCutPlusModular: public GeneralizedCutOracle<ValueType> {
public:
  STCutPlusModular(): GeneralizedCutOracle<ValueType>() {}

  std::string GetName() { return "S-T Cut + Modular"; }

  static STCutPlusModular<ValueType> FromEdgeList(
    std::size_t n, element_type s, element_type t,
    const std::vector<std::pair<element_type, element_type>>& edges,
    const std::vector<ValueType>& capacities,
    const std::vector<ValueType>& modular_term
  );

};

/*
template <typename ValueType>
class Cut: GeneralizedCutOracle<ValueType> {
public:
  Cut(): GeneralizedCutOracle<ValueType>() {}

  static Cut<ValueType> FromEdgeList(
    std::size_t n, DirectionKind directed,
    const std::vector<std::pair<element_type, element_type>>& edges,
    const std::vector<ValueType>& capacities
  );

};
*/

template <typename ValueType>
class CutPlusModular: public GeneralizedCutOracle<ValueType> {
public:
  CutPlusModular(): GeneralizedCutOracle<ValueType>() {}

  std::string GetName() { return "Cut + Modular"; }

  static CutPlusModular<ValueType> FromEdgeList(
    std::size_t n, DirectionKind directed,
    const std::vector<std::pair<element_type, element_type>>& edges,
    const std::vector<ValueType>& capacities,
    const std::vector<ValueType>& modular_term
  );

};


template <typename ValueType>
STCut<ValueType> STCut<ValueType>::FromEdgeList(
  std::size_t n, element_type s, element_type t,
  const std::vector<std::pair<element_type, element_type>>& edges,
  const std::vector<ValueType>& capacities
)
{
  MaxflowGraph<ValueType> graph;
  auto m = edges.size();
  if (m != capacities.size()) {
    throw std::invalid_argument("STCut::FromEdgeList: edges.size() and capacities.size() do not match");
  }
  if (s >= n + 2 || t >= n + 2) {
    throw std::invalid_argument("STCut::FromEdgeList: Node names should be less than n + 2");
  }
  graph.Reserve(n + 2, edges.size());
  for (element_type i = 0; i < n + 2; ++i) {
    graph.AddNode(i); // Add n + 2 nodes (n inners, source and sink nodes)
  }
  auto source = graph.GetNodeById(s);
  auto sink = graph.GetNodeById(t);

  for (std::size_t edge_id = 0; edge_id < m; ++edge_id) {
    auto src = edges[edge_id].first;
    auto dst = edges[edge_id].second;
    if (src >= n + 2 || dst >= n + 2) {
      throw std::invalid_argument("STCut::FromEdgeList: Node names should be less than n + 2");
    }
    auto head = graph.GetNodeById(dst); //NOTE: In this case, node->index == node->name is true for all nodes
    auto tail = graph.GetNodeById(src);
    ValueType cap = capacities[edge_id];
    ValueType cap_rev = 0;
    if (src == s && dst != t) {
      graph.AddSVArcPair(head, tail, cap, cap_rev);
    }
    else if (src != s && dst == t) {
      graph.AddVTArcPair(head, tail, cap, cap_rev);
    }
    else if (src != s && dst != t) {
      if (cap < 0) {
        throw std::invalid_argument("STCut::FromEdgeList: Negative inner capacity");
      }
      graph.AddArcPair(head, tail, cap, cap_rev);
    }
    else {
      graph.AddSTOffset(cap);
    }
  }
  graph.MakeGraph(source, sink);
  STCut<ValueType> F;
  F.SetGraph(std::move(graph));
  return F;
}

template <typename ValueType>
STCutPlusModular<ValueType> STCutPlusModular<ValueType>::FromEdgeList(
  std::size_t n, element_type s, element_type t,
  const std::vector<std::pair<element_type, element_type>>& edges,
  const std::vector<ValueType>& capacities,
  const std::vector<ValueType>& modular_term
)
{
  MaxflowGraph<ValueType> graph;
  auto m = edges.size();
  if (m != capacities.size()) {
    throw std::invalid_argument("STCutPlusModular::FromEdgeList: edges.size() and capacities.size() do not match");
  }
  if (modular_term.size() != n + 2) {
    // NOTE: modular_term[s] and modular_term[t] will be ignored
    throw std::invalid_argument("STCutPlusModular::FromEdgeList: modular_term.size() should be n + 2");
  }
  if (s >= n + 2 || t >= n + 2) {
    throw std::invalid_argument("STCutPlusModular::FromEdgeList: Node names should be less than n + 2");
  }
  graph.Reserve(n + 2, edges.size());
  for (element_type i = 0; i < n + 2; ++i) {
    graph.AddNode(i);
  }
  auto source = graph.GetNodeById(s);
  auto sink = graph.GetNodeById(t);

  for (std::size_t node_id = 0; node_id < n + 2; ++node_id) {
    if (node_id != s && node_id != t) {
      auto node = graph.GetNodeById(node_id);
      auto xi = modular_term[node_id];
      graph.AddVTArcPair(sink, node, xi, 0);
      graph.AddSVArcPair(node, source, 0, 0);
    }
  }

  for (std::size_t edge_id = 0; edge_id < m; ++edge_id) {
    auto src = edges[edge_id].first;
    auto dst = edges[edge_id].second;
    if (src >= n || dst >= n) {
      throw std::invalid_argument("STCutPlusModular::FromEdgeList: Node names should be less than n + 2");
    }
    auto head = graph.GetNodeById(dst);
    auto tail = graph.GetNodeById(src);
    ValueType cap = capacities[edge_id];
    ValueType cap_rev = 0;
    if (src == s && dst != t) {
      graph.AddSVArcPair(head, tail, cap, cap_rev);
    }
    else if (src != s && dst == t) {
      graph.AddVTArcPair(head, tail, cap, cap_rev);
    }
    else if (src != s && dst != t) {
      if (cap < 0) {
        throw std::invalid_argument("STCut::FromEdgeList: Negative inner capacity");
      }
      graph.AddArcPair(head, tail, cap, cap_rev);
    }
    else {
      graph.AddSTOffset(cap);
    }
  }
  graph.MakeGraph(source, sink);
  STCutPlusModular<ValueType> F;
  F.SetGraph(std::move(graph));
  return F;
}

template <typename ValueType>
CutPlusModular<ValueType> CutPlusModular<ValueType>::FromEdgeList(
  std::size_t n, DirectionKind directed,
  const std::vector<std::pair<element_type, element_type>>& edges,
  const std::vector<ValueType>& capacities,
  const std::vector<ValueType>& modular_term
)
{
  MaxflowGraph<ValueType> graph;
  auto m = edges.size();
  if (m != capacities.size()) {
    throw std::invalid_argument("CutPlusModular::FromEdgeList: edges.size() and capacities.size() do not match");
  }
  if (modular_term.size() != n) {
    throw std::invalid_argument("CutPlusModular::FromEdgeList: modular_term.size() should be n");
  }
  graph.Reserve(n + 2, m);
  std::size_t s = n;
  std::size_t t = n + 1;
  for (element_type i = 0; i < n + 2; ++i) {
    graph.AddNode(i); // Add n + 2 nodes (n inners, source and sink nodes)
  }
  auto source = graph.GetNodeById(s);
  auto sink = graph.GetNodeById(t);

  for (std::size_t node_id = 0; node_id < n; ++node_id) {
    // NOTE: We don't have to worry about s-v or v-t capacities being non-positive, because they are
    // corrected by the maxflow algorithm.
    auto node = graph.GetNodeById(node_id);
    auto xi = modular_term[node_id];
    graph.AddVTArcPair(sink, node, xi, 0);
    graph.AddSVArcPair(node, source, 0, 0);
  }

  for (std::size_t edge_id = 0; edge_id < m; ++edge_id) {
    auto src = edges[edge_id].first;
    auto dst = edges[edge_id].second;
    if (src >= n || dst >= n) {
      throw std::invalid_argument("STCut::FromEdgeList: Node names should be less than n");
    }
    auto head = graph.GetNodeById(dst); //NOTE: In this case, node->index == node->name is true for all nodes
    auto tail = graph.GetNodeById(src);
    ValueType cap = capacities[edge_id];
    ValueType cap_rev = (directed == DIRECTED) ? 0 : cap;
    if (cap < 0) {
      throw std::invalid_argument("STCut::FromEdgeList: Negative inner capacity");
    }
    graph.AddArcPair(head, tail, cap, cap_rev);
  }
  graph.MakeGraph(source, sink);
  CutPlusModular<ValueType> F;
  F.SetGraph(std::move(graph));
  return F;
}


}

#endif
