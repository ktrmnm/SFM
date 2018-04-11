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

#ifndef MAXFLOW_GRAPH_H
#define MAXFLOW_GRAPH_H

#include <algorithm>
#include <cstddef>
#include <deque>
#include <iterator>
#include <limits>
#include <list>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/graph.h"
#include "core/utils.h"
#include "core/set_utils.h"

//#define DEBUG
#ifdef DEBUG
#include <iostream>
#endif

namespace submodular {

template <typename ValueType> struct NodeM;
template <typename ValueType> struct ArcM;
template <typename ValueType> class MaxflowGraph;
template <typename ValueType> struct MaxflowState;
struct MaxflowStats;


template <typename ValueType>
struct NodeM {
  using arc_type = Arc<ValueType>;
  using value_type = ValueType;
  using color_type = NodeColor;
  using Node_s = std::shared_ptr<NodeM<ValueType>>;
  using Arc_s = std::shared_ptr<ArcM<ValueType>>;
  using Arc_w = std::weak_ptr<ArcM<ValueType>>;

  element_type name;
  std::size_t index; // index in vector
  std::size_t typed_index;
  bool is_variable;
  value_type excess;
  Height_t height;
  NodeColor color;

  Arc_w sv_arc; //! reference to arc from the "representative source" node
  value_type sv_cap; //! capacity of sv_arc (can be negative)
  Arc_s GetSVArc() { return sv_arc.lock(); }

  Arc_w vt_arc; //! reference to arc to the "representative sink" node
  value_type vt_cap; //! capacity of vt_cap (can be negative)
  Arc_s GetVTArc() { return vt_arc.lock(); }

  typename std::vector<Arc_s>::iterator current_arc; //! position of current arc in adjacency list
  typename std::list<Node_s>::iterator pos_in_layer; //! position of node in inactive_layer_
};

template <typename ValueType>
struct ArcM {
  using node_type = NodeM<ValueType>;
  using value_type = ValueType;
  using Node_s = typename NodeTraits<node_type>::type_s;
  using Node_w = typename NodeTraits<node_type>::type_w;
  using Arc_s  = typename ArcTraits<ArcM<ValueType>>::type_s;
  using Arc_w  = typename ArcTraits<ArcM<ValueType>>::type_w;

  value_type flow;
  value_type capacity;
  Arc_w reversed;
  Node_w head_node;
  Node_w tail_node;
  std::size_t hash;

  value_type GetResidual() { return capacity - flow; }
  Arc_s GetReversed() { return reversed.lock(); }
  Node_s GetHeadNode() { return head_node.lock(); }
  Node_s GetTailNode() { return tail_node.lock(); }
};

template <typename ValueType>
struct MaxflowState {
  std::size_t first_inner_index;
  std::size_t first_sink_index;
  std::vector<std::size_t> first_alive_arc_index;
  ValueType alpha;
  ValueType st_offset; //! capacity of s->t arc. (can be negative)
};


template <typename ValueType>
class MaxflowGraph {
public:
  using node_type = NodeM<ValueType>;
  using arc_type = ArcM<ValueType>;
  using value_type = ValueType;
  using state_type = MaxflowState<ValueType>;
  using Node_s = typename NodeTraits<node_type>::type_s;
  using Node_w = typename NodeTraits<node_type>::type_w;
  using Arc_s = typename ArcTraits<arc_type>::type_s;
  using Arc_w = typename ArcTraits<arc_type>::type_w;

  using node_iterator = typename std::vector<Node_s>::iterator;
  using c_node_iterator = const typename std::vector<Node_s>::iterator;
  using arc_iterator = typename std::vector<Arc_s>::iterator;
  using c_arc_iterator = const typename std::vector<Arc_s>::iterator;

  MaxflowGraph();
  MaxflowGraph(const MaxflowGraph&) = default;
  MaxflowGraph(MaxflowGraph&&) = default;
  MaxflowGraph& operator=(const MaxflowGraph&) = default;
  MaxflowGraph& operator=(MaxflowGraph&&) = default;

  // Methods for graph construction
  void Reserve(std::size_t n, std::size_t m);
  Node_s AddNode(element_type name, bool is_variable = true);
  Arc_s AddArc(const Node_s& head, const Node_s& tail, value_type cap);
  Arc_s AddSVArc(const Node_s& head, const Node_s& tail, value_type cap);
  Arc_s AddVTArc(const Node_s& head, const Node_s& tail, value_type cap);
  void AddArcPair(const Node_s& head, const Node_s& tail, value_type cap, value_type rev_cap);
  void AddSVArcPair(const Node_s& head, const Node_s& tail, value_type cap, value_type rev_cap);
  void AddVTArcPair(const Node_s& head, const Node_s& tail, value_type cap, value_type rev_cap);

  void AddSTOffset(value_type cap);
  value_type GetSTOffset() const;
  void MakeGraph(const Node_s& source, const Node_s& sink);
  void AddCardinalityFunction(value_type multiplier);
  void SetTol(value_type tol) { tol_ = tol; }

  void SetNGround(std::size_t n_ground) { n_ground_ = n_ground; }
  std::size_t GetNGround() const { return n_ground_; }

  // Methods to get graph information
  std::size_t GetNodeNumber() const;
  bool HasNode(element_type name) const;
  bool HasAuxiliaryNodes() const { return has_auxiliary_nodes_; }
  Node_s GetNode(element_type name) const;
  Node_s GetNodeById(std::size_t index) const;
  Node_s GetSourceNode() const;
  Node_s GetSinkNode() const;
  bool IsInnerNode(const Node_s& node) const;
  bool IsSourceNode(const Node_s& node) const;
  bool IsSinkNode(const Node_s& node) const;
  value_type GetArcBaseCap(const Arc_s& arc) const;
  std::vector<std::size_t> GetInnerIndices(bool filter_variable = false) const;
  std::vector<element_type> GetMembers() const;
  std::size_t Id2Name(std::size_t index) const { return GetNodeById(index)->name; }

  // Maxflow utilities
  void InitPreflowPush();
  bool FindMaxPreflow(unsigned int max_work_amount);
  void FindMaxPreflow();
  value_type GetMaxFlowValue();

  // Mincut utilities
  value_type GetCutValueByIds(const std::vector<std::size_t>& node_ids);
  value_type GetCutValueByNames(const std::vector<element_type>& members);
  value_type GetCutValue(const std::vector<TermType>& cut);
  void FindMinCut();
  TermType WhatSegment(std::size_t node_id);
  std::vector<std::size_t> GetMinCut(bool filter_variable = false);

  // Reduction / contraction utilities
  MaxflowState<ValueType> GetState() const;
  void RestoreState(MaxflowState<ValueType> state);

  void Reduction(const std::vector<TermType>& cut);
  void ReductionByIds(const std::vector<std::size_t>& node_ids);
  void ReductionByNames(const std::vector<element_type>& members);
  void Contraction(const std::vector<TermType>& cut, value_type additional_offset = 0);
  void ContractionByIds(const std::vector<std::size_t>& node_ids, value_type additional_offset = 0);
  void ContractionByNames(const std::vector<element_type>& members, value_type additional_offset = 0);

private:
  std::size_t n_ground_;
  std::vector<Node_s> nodes_;
  std::unordered_map<element_type, std::size_t> name2id_; // convert node names to indices in nodes_
  std::vector<Arc_s> arcs_;
  std::vector<std::vector<Arc_s>> adj_;
  bool has_auxiliary_nodes_;

  std::vector<TermType> mincut_;
  value_type flow_offset_;
  Height_t max_height_;
  MaxflowState<ValueType> state_;
  std::vector<Node_s> typed_node_list_;
  std::vector<std::list<Node_s>> active_layer_;
  std::vector<std::list<Node_s>> inactive_layer_;

  //MaxflowStats stats_;
  bool done_max_preflow_;
  bool done_mincut_;

  struct MaxflowStats {
    unsigned int work_counter;
    unsigned int work_since_last_update;
    void Clear() { work_counter = 0; work_since_last_update = 0; }
  };
  MaxflowStats stats_;

  value_type tol_;

  void InitHeights();
  void ClearFlow(const Node_s& node);
  void InitCaps();

  // for highest-first data structure
  void InitLayers();
  void AddToActiveLayer(const Node_s& node);
  bool ActiveNodeExists(Height_t height) const;
  Node_s PopActiveNode(Height_t height);
  void AddToInactiveLayer(const Node_s& node);
  bool InactiveNodeExists(Height_t height);
  void RemoveFromInactiveLayer(const Node_s& node);

  // for reduction & contraction
  void MakeSource(const Node_s& node);
  void UnmakeSource(const Node_s& node);
  void MakeSink(const Node_s& node);
  void UnmakeSink(const Node_s& node);

  bool IsArcAlive(const Arc_s& arc) const;
  void _RearrangeAliveArcs(std::size_t node_id);
  void RearrangeAliveArcs();

  // subroutines for preflow-push algorithm
  void Push(const Arc_s& arc, value_type amount);
  Height_t Relabel(const Node_s& node);
  void Discharge(const Node_s& node);
  void GlobalRelabeling();
  void Gap(Height_t height);

  auto AliveArcBegin(std::size_t node_id) {
    return std::next(adj_[node_id].begin(), state_.first_alive_arc_index[node_id]);
  }
  auto AliveArcEnd(std::size_t node_id) {
    return adj_[node_id].end();
  }
  // TODO: 現状の用途ではconst指定してもたまたまコンパイルが通るが、ちゃんと整理する。
  auto InnerBegin() const {
    return std::next(typed_node_list_.begin(), state_.first_inner_index);
  }
  auto InnerEnd() const {
    return std::next(typed_node_list_.begin(), state_.first_sink_index);
  }

  // Tuning parameters to control the frequency of the global relabeling
  // heuristic. At each relabeling step, the work counter is incremented by
  // relabel_work_const_ + relabel_work_per_arc_ * #{updated arcs}.
  // Global relabeling is executed if the work counter is increased by the
  // following amount since the last update: (alpha_n * nodes_.size() + alpha_m
  // * arcs_.size()) * global_relabel_period_
  static const unsigned int global_relabel_period_ = 2;
  static const unsigned int relabel_work_const_ = 12;
  static const unsigned int relabel_work_per_arc_ = 1;
  static const unsigned int alpha_n_ = 6;
  static const unsigned int alpha_m_ = 1;
};

template <typename ValueType>
MaxflowGraph<ValueType>::MaxflowGraph()
  : n_ground_(0),
    has_auxiliary_nodes_(false),
    flow_offset_(0),
    done_max_preflow_(false),
    done_mincut_(false),
    tol_(1e-8)
{
  state_.alpha = 0;
  state_.st_offset = 0;
  stats_.work_counter = 0;
  stats_.work_since_last_update = 0;
}

template <typename ValueType>
void MaxflowGraph<ValueType>::Reserve(std::size_t n, std::size_t m) {
  nodes_.reserve(n);
  arcs_.reserve(m);
  typed_node_list_.reserve(n);
}

template <typename ValueType>
typename MaxflowGraph<ValueType>::Node_s
MaxflowGraph<ValueType>::AddNode(element_type name, bool is_variable) {
  if (!is_variable && !has_auxiliary_nodes_) {
    has_auxiliary_nodes_ = true;
  }
  if (name2id_.count(name) == 1) {
    auto node = nodes_[name2id_[name]];
    node->is_variable = is_variable;
    return node;
  }
  else {
    auto node = std::make_shared<NodeM<ValueType>>();
    node->excess = 0;
    node->height = -1;
    node->sv_cap = 0;
    node->vt_cap = 0;
    node->name = name;
    node->index = nodes_.size();
    node->typed_index = 0;
    node->is_variable = is_variable;
    nodes_.push_back(node);

    name2id_.insert({ node->name, node->index });
    if (is_variable) {
      n_ground_ = std::max((std::size_t)name + 1, n_ground_);
    }

    return node;
  }
}

template <typename ValueType>
typename MaxflowGraph<ValueType>::Arc_s
MaxflowGraph<ValueType>::AddArc(const Node_s& head, const Node_s& tail, value_type cap) {
  auto arc = std::make_shared<ArcM<ValueType>>();
  arc->flow = 0;
  arc->capacity = cap;
  arc->head_node = head;
  arc->tail_node = tail;
  arcs_.push_back(arc);
  return arc;
}

template <typename ValueType>
typename MaxflowGraph<ValueType>::Arc_s
MaxflowGraph<ValueType>::AddSVArc(const Node_s& head, const Node_s& tail, value_type cap) {
  if (!(head->sv_arc).expired()) {
    auto sv_arc = head->GetSVArc();
    sv_arc->capacity += cap;
    head->sv_cap += cap;
    return sv_arc;
  }
  else {
    auto arc = AddArc(head, tail, cap);
    head->sv_arc = arc;
    head->sv_cap = cap;
    return arc;
  }
}

template <typename ValueType>
typename MaxflowGraph<ValueType>::Arc_s
MaxflowGraph<ValueType>::AddVTArc(const Node_s& head, const Node_s& tail, value_type cap) {
  if (!(tail->vt_arc).expired()) {
    auto vt_arc = tail->GetVTArc();
    vt_arc->capacity += cap;
    tail->vt_cap += cap;
    return vt_arc;
  }
  else {
    auto arc = AddArc(head, tail, cap);
    tail->vt_arc = arc;
    tail->vt_cap = cap;
    return arc;
  }
}

template <typename ValueType>
void MaxflowGraph<ValueType>::AddArcPair(const Node_s& head, const Node_s& tail, value_type cap, value_type rev_cap) {
  auto arc = AddArc(head, tail, cap);
  auto rev = AddArc(tail, head, rev_cap);
  arc->reversed = rev;
  rev->reversed = arc;
}

template <typename ValueType>
void MaxflowGraph<ValueType>::AddSVArcPair(const Node_s& head, const Node_s& tail, value_type cap, value_type rev_cap) {
  auto arc = AddSVArc(head, tail, cap);
  auto rev = AddArc(tail, head, rev_cap);
  arc->reversed = rev;
  rev->reversed = arc;
}

template <typename ValueType>
void MaxflowGraph<ValueType>::AddVTArcPair(const Node_s& head, const Node_s& tail, value_type cap, value_type rev_cap) {
  auto arc = AddVTArc(head, tail, cap);
  auto rev = AddArc(tail, head, rev_cap);
  arc->reversed = rev;
  rev->reversed = arc;
}

template <typename ValueType>
void MaxflowGraph<ValueType>::AddSTOffset(value_type cap) {
  state_.st_offset += cap;
}

template <typename ValueType>
typename MaxflowGraph<ValueType>::value_type MaxflowGraph<ValueType>::GetSTOffset() const {
  return state_.st_offset;
}

template <typename ValueType>
void MaxflowGraph<ValueType>::MakeGraph(const Node_s& source, const Node_s& sink) {
  // 1. prepare typed_node_list_
  typed_node_list_.clear();
  typed_node_list_.reserve(nodes_.size());
  for (const auto& node : nodes_) {
    node->typed_index = typed_node_list_.size();
    typed_node_list_.push_back(node);
  }

  state_.first_inner_index = 1;
  auto node0 = typed_node_list_[0];
  std::swap(typed_node_list_[0], typed_node_list_[source->typed_index]);
  node0->typed_index = source->typed_index;
  source->typed_index = 0;

  state_.first_sink_index = (int)typed_node_list_.size() - 1;
  auto node1 = typed_node_list_[state_.first_sink_index];
  std::swap(typed_node_list_[state_.first_sink_index],
            typed_node_list_[sink->typed_index]);
  node1->typed_index = sink->typed_index;
  sink->typed_index = state_.first_sink_index;

  // add s->v and v->t arcs if they do not exist
  for (auto it = InnerBegin(); it != InnerEnd(); ++it) {
    auto node = *it;
    if ((node->sv_arc).expired()) {
      auto source = GetSourceNode();
      AddSVArcPair(node, source, 0, 0);
    }
    if ((node->vt_arc).expired()) {
      auto sink = GetSinkNode();
      AddVTArcPair(sink, node, 0, 0);
    }
  }

  // 2 make adjacency list
  adj_.clear();
  adj_.resize(nodes_.size());
  state_.first_alive_arc_index.clear();
  state_.first_alive_arc_index.resize(nodes_.size(), 0);
  for (const auto& arc : arcs_) {
    std::size_t tail_id = arc->GetTailNode()->index;
    adj_[tail_id].push_back(arc);
  }
  for (const auto &node : nodes_) {
    node->current_arc = adj_[node->index].begin();
  }
}

template <typename ValueType>
std::size_t MaxflowGraph<ValueType>::GetNodeNumber() const {
  return state_.first_sink_index - state_.first_inner_index + 2;
}

template <typename ValueType>
bool MaxflowGraph<ValueType>::HasNode(element_type name) const {
  return name2id_.count(name) == 1;
}

template <typename ValueType>
typename MaxflowGraph<ValueType>::Node_s MaxflowGraph<ValueType>::GetNode(element_type name) const {
  return HasNode(name) ? nodes_[name2id_.at(name)] : nullptr;
}

template <typename ValueType>
typename MaxflowGraph<ValueType>::Node_s MaxflowGraph<ValueType>::GetNodeById(std::size_t index) const {
  return nodes_[index];
}

template <typename ValueType>
typename MaxflowGraph<ValueType>::Node_s MaxflowGraph<ValueType>::GetSourceNode() const {
  return typed_node_list_.front();
}

template <typename ValueType>
typename MaxflowGraph<ValueType>::Node_s MaxflowGraph<ValueType>::GetSinkNode() const {
  return typed_node_list_.back();
}

template <typename ValueType>
bool MaxflowGraph<ValueType>::IsInnerNode(const Node_s& node) const {
  return node->typed_index >= state_.first_inner_index &&
         node->typed_index < state_.first_sink_index;
}

template <typename ValueType>
bool MaxflowGraph<ValueType>::IsSourceNode(const Node_s& node) const {
  return node->typed_index < state_.first_inner_index;
}

template <typename ValueType>
bool MaxflowGraph<ValueType>::IsSinkNode(const Node_s& node) const {
  return node->typed_index >= state_.first_sink_index;
}

template <typename ValueType>
void MaxflowGraph<ValueType>::InitLayers() {
  active_layer_.clear();
  inactive_layer_.clear();
  active_layer_.resize(GetNodeNumber());
  inactive_layer_.resize(GetNodeNumber());
}

template <typename ValueType>
void MaxflowGraph<ValueType>::AddToActiveLayer(const Node_s& node) {
  Height_t height = node->height;
  active_layer_[height].push_front(node);
  max_height_ = std::max(max_height_, height);
}

template <typename ValueType>
bool MaxflowGraph<ValueType>::ActiveNodeExists(Height_t height) const {
  if (height >= (int)GetNodeNumber()) {
    return false;
  } else {
    return !active_layer_[height].empty();
  }
}

template <typename ValueType>
typename MaxflowGraph<ValueType>::Node_s MaxflowGraph<ValueType>::PopActiveNode(Height_t height) {
  Node_s node = active_layer_[height].front();
  active_layer_[height].pop_front();
  return node;
}

template <typename ValueType>
void MaxflowGraph<ValueType>::AddToInactiveLayer(const Node_s& node) {
  Height_t height = node->height;
  inactive_layer_[height].push_front(node);
  node->pos_in_layer = inactive_layer_[height].begin();
}

template <typename ValueType>
bool MaxflowGraph<ValueType>::InactiveNodeExists(Height_t height) {
  if (height >= (Height_t)GetNodeNumber()) {
    return false;
  } else {
    return !inactive_layer_[height].empty();
  }
}

template <typename ValueType>
void MaxflowGraph<ValueType>::RemoveFromInactiveLayer(const Node_s& node) {
  Height_t height = node->height;
  inactive_layer_[height].erase(node->pos_in_layer);
}

template <typename ValueType>
void MaxflowGraph<ValueType>::AddCardinalityFunction(value_type multiplier) {
  for (auto it = InnerBegin(); it != InnerEnd(); ++it) {
    auto node = *it;
    if (node->is_variable) {
      node->vt_cap += multiplier;
    }
  }
}

template <typename ValueType>
void MaxflowGraph<ValueType>::InitHeights() {
  GetSourceNode()->height = (int) GetNodeNumber();
  GetSinkNode()->height = 0;
  for (auto it = InnerBegin(); it != InnerEnd(); ++it) {
    auto node = *it;
    node->height = 1;
    AddToInactiveLayer(node);
    node->current_arc = AliveArcBegin(node->index);
  }
  max_height_ = 1;
}

template <typename ValueType>
void MaxflowGraph<ValueType>::ClearFlow(const Node_s& node) {
  node->excess = 0;
  for (auto it = AliveArcBegin(node->index); it != AliveArcEnd(node->index); ++it) {
    auto arc = *it;
    arc->flow = 0;
    arc->GetReversed()->flow = 0;
  }
}

// Update capacities of s->v (v->t) arcs by sv_cap (resp. vt_cap)
template <typename ValueType>
void MaxflowGraph<ValueType>::InitCaps() {
  value_type delta = 0;
  for (auto it = InnerBegin(); it != InnerEnd(); ++it) {
    auto node = *it;
    value_type delta_new = std::min(node->sv_cap, node->vt_cap);
    if (delta_new < delta) {
      delta = delta_new;
    }
  }

  flow_offset_ = 0;
  for (auto it = InnerBegin(); it != InnerEnd(); ++it) {
    auto node = *it;
    (node->GetSVArc())->capacity = (node->sv_cap) - delta;
    (node->GetVTArc())->capacity = (node->vt_cap) - delta;
    flow_offset_ += delta;
  }

}

template <typename ValueType>
void MaxflowGraph<ValueType>::InitPreflowPush() {
  #ifdef DEBUG
  std::cout << "InitPreflowPush" << std::endl;
  #endif
  done_max_preflow_ = false;
  done_mincut_ = false;

  auto source = GetSourceNode();
  auto sink = GetSinkNode();
  ClearFlow(source);
  ClearFlow(sink);
  for (auto it = InnerBegin(); it != InnerEnd(); ++it) {
    auto node = *it;
    ClearFlow(node);
  }

  InitCaps();
  InitLayers();
  InitHeights();

  for (auto it = AliveArcBegin(source->index); it != AliveArcEnd(source->index); ++it) {
    auto arc = *it;
    value_type res = arc->GetResidual();
    if (res > 0 && !utils::is_abs_close(res, value_type(0), tol_)) {
      Push(arc, res);
    }
  }

  stats_.Clear();

}

template <typename ValueType>
bool MaxflowGraph<ValueType>::FindMaxPreflow(unsigned int max_work_amount) {
  #ifdef DEBUG
  std::cout << "FindMaxPreflow" << std::endl;
  #endif
  unsigned int global_relabel_barrier =
      (alpha_n_ * nodes_.size() + alpha_m_ * arcs_.size()) *
      global_relabel_period_;

  while (stats_.work_counter < max_work_amount) {

    if (stats_.work_since_last_update > global_relabel_barrier) {
      GlobalRelabeling();
      stats_.work_since_last_update = 0;
    }

    while (max_height_ >= 0 && !ActiveNodeExists(max_height_)) {
      --max_height_;
    }
    if (max_height_ < 0) {
      done_max_preflow_ = true;
      return true;
    }

    Node_s node = PopActiveNode(max_height_);

    Discharge(node);

    stats_.work_counter += stats_.work_since_last_update;
  }
  done_max_preflow_ = true;
  return false;
}

template <typename ValueType>
void MaxflowGraph<ValueType>::FindMaxPreflow() {
  FindMaxPreflow(std::numeric_limits<unsigned int>::max());
}

template <typename ValueType>
typename MaxflowGraph<ValueType>::value_type MaxflowGraph<ValueType>::GetMaxFlowValue() {
  if (!done_max_preflow_) {
    InitPreflowPush();
    FindMaxPreflow();
  }
  auto sink_excess = GetSinkNode()->excess;
  return sink_excess + flow_offset_ + state_.st_offset;
}

template <typename ValueType>
void MaxflowGraph<ValueType>::Push(const Arc_s& arc, value_type amount) {
  arc->flow += amount;
  arc->GetReversed()->flow -= amount;
  arc->GetTailNode()->excess -= amount;

  Node_s head = arc->GetHeadNode();
  if (IsInnerNode(head) && utils::is_abs_close(head->excess, value_type(0), tol_)) {
    RemoveFromInactiveLayer(head);
    AddToActiveLayer(head);
  }
  head->excess += amount;
}

template <typename ValueType>
Height_t MaxflowGraph<ValueType>::Relabel(const Node_s& node) {
  stats_.work_since_last_update += relabel_work_const_;

  std::size_t index = node->index;
  Height_t min_height = 2 * GetNodeNumber();

  auto min_arc_it = AliveArcBegin(index);
  for (auto it = AliveArcBegin(index); it != AliveArcEnd(index); ++it) {
    stats_.work_since_last_update += relabel_work_per_arc_;
    auto arc = *it;
    Height_t current_height = arc->GetHeadNode()->height;
    value_type res = arc->GetResidual();
    if (res > 0 && !utils::is_abs_close(res, value_type(0), tol_) && min_height > current_height) {
      min_height = current_height;
      min_arc_it = it;
    }
  }

  node->current_arc = min_arc_it;
  node->height = min_height + 1;
  return node->height;
}

template <typename ValueType>
void MaxflowGraph<ValueType>::Gap(Height_t height) {
  Height_t n = GetNodeNumber();
  for (Height_t h = height; h <= max_height_; ++h) {
    // clear active nodes
    for (const auto &node : active_layer_[h]) {
      node->height = n;
    }
    active_layer_[h].clear();

    // clear inactive nodes
    for (const auto &node : inactive_layer_[h]) {
      node->height = n;
    }
    inactive_layer_[h].clear();
  }
  max_height_ = height - 1;
}

template <typename ValueType>
void MaxflowGraph<ValueType>::Discharge(const Node_s& node) {
  #ifdef DEBUG
  std::cout << "Discharging node " << node->index << std::endl;
  #endif
  Height_t n = GetNodeNumber();

  while (true) {
    Arc_s current_arc = *(node->current_arc);
    value_type res = current_arc->GetResidual();

    if (res > 0 && !utils::is_abs_close(res, value_type(0), tol_)) {
      auto head = current_arc->GetHeadNode();

      // push if admissible
      if (head->height + 1 == node->height) {
        value_type update = std::min(node->excess, res);
        Push(current_arc, update);
        if (utils::is_abs_close(node->excess, value_type(0), tol_)) {
          break;
        }
      }
    }

    if (node->current_arc != std::prev(AliveArcEnd(node->index))) {
      node->current_arc++;
    }
    else if (node->current_arc == std::prev(AliveArcEnd(node->index))) {
      // If the current edge is the last edge in the adjacency list,
      // then try one of the followings:
      // 1. gap heuristics
      // 2. relabel node and make a admissible edge.
      Height_t old_height = node->height;
      if (!ActiveNodeExists(old_height) && !InactiveNodeExists(old_height)) {
        Gap(old_height);
        node->height = n;
        break;
      }

      auto new_height = Relabel(node);

      // Note: In the preflow maximizing phase (aka "first phase"),
      // it is known that a node is non-reachable from the sink when
      // node->height >= n. We can therefore stop the discharge operation.
      if (new_height >= n) {
        break;
      }
    }
  }

  if (node->height < n) {
    if (node->excess > 0 && !utils::is_abs_close(node->excess, value_type(0), tol_)) {
      AddToActiveLayer(node);
    } else {
      AddToInactiveLayer(node);
    }
  }
}

// Update distance labels (i.e. heights) by BFS
template <typename ValueType>
void MaxflowGraph<ValueType>::GlobalRelabeling() {
  InitLayers();
  Height_t n = GetNodeNumber();

  std::deque<Node_s> Q;
  for (auto &node : nodes_) {
    node->color = WHITE;
    node->height = n;
  }
  auto sink = GetSinkNode();
  sink->color = BLACK;
  sink->height = 0;
  Q.push_back(sink);

  while (!Q.empty()) { // BFS start
    auto node = Q.front();
    Q.pop_front();

    Height_t next_height = node->height + 1;

    for (auto it = AliveArcBegin(node->index); it != AliveArcEnd(node->index); ++it) {
      auto arc = *it;
      auto rev_arc = arc->GetReversed();
      value_type rev_res = rev_arc->GetResidual();

      if (rev_res > 0 && !utils::is_abs_close(rev_res, value_type(0), tol_)) {
        auto next_node = arc->GetHeadNode();
        if (next_node->color == WHITE) {
          next_node->color = BLACK;
          next_node->height = next_height;

          if (next_node->excess > 0 && !utils::is_abs_close(next_node->excess, value_type(0), tol_)) {
            AddToActiveLayer(next_node);
          } else {
            AddToInactiveLayer(next_node);
          }
          Q.push_back(next_node);
        }
      }
    }
  } // BFS
}

template <typename ValueType>
bool MaxflowGraph<ValueType>::IsArcAlive(const Arc_s& arc) const {
  auto head = arc->GetHeadNode();
  auto tail = arc->GetTailNode();
  bool tail_is_inner = IsInnerNode(tail);
  bool head_is_inner = IsInnerNode(head);
  auto source_to_tail = tail->GetSVArc();
  auto tail_to_sink = tail->GetVTArc();
  auto source_to_head = head->GetSVArc();
  auto head_to_sink = head->GetVTArc();
  auto reversed = arc->GetReversed();

  return (tail_is_inner && head_is_inner) ||
         (tail_is_inner && arc == tail_to_sink) ||
         (tail_is_inner && reversed == source_to_tail) ||
         (head_is_inner && arc == source_to_head) ||
         (head_is_inner && reversed == head_to_sink);
}

template <typename ValueType>
typename MaxflowGraph<ValueType>::value_type MaxflowGraph<ValueType>::GetArcBaseCap(const Arc_s& arc) const {
  auto head = arc->GetHeadNode();
  auto tail = arc->GetTailNode();
  if (arc == head->GetSVArc()) {
    return head->sv_cap;
  } else if (arc == tail->GetVTArc()) {
    return tail->vt_cap;
  } else {
    return arc->capacity;
  }
}

template <typename ValueType>
void MaxflowGraph<ValueType>::FindMinCut() {
  if (!done_max_preflow_) {
    InitPreflowPush();
    FindMaxPreflow();
  }

  mincut_.clear();
  mincut_.resize(nodes_.size(), SOURCE);

  std::deque<Node_s> Q;
  GetSourceNode()->color = WHITE;
  for (auto it = InnerBegin(); it != InnerEnd(); ++it) {
    auto node = *it;
    node->color = WHITE;
  }
  auto sink = GetSinkNode();
  sink->color = BLACK;
  mincut_[sink->index] = SINK;
  Q.push_back(sink);

  while (!Q.empty()) { // BFS start
    auto node = Q.front();
    Q.pop_front();
    for (auto it = AliveArcBegin(node->index); it != AliveArcEnd(node->index); ++it) {
      auto rev_arc = (*it)->GetReversed();
      value_type res = rev_arc->GetResidual();
      if (res > 0 && !utils::is_abs_close(res, value_type(0), tol_)) {
        auto dst = rev_arc->GetTailNode();
        if (dst->color == WHITE) {
          dst->color = BLACK;
          mincut_[dst->index] = SINK;
          Q.push_back(dst);
        }
      }
    }
  }// BFS
  done_mincut_ = true;
}

template <typename ValueType>
TermType MaxflowGraph<ValueType>::WhatSegment(std::size_t node_id) {
  if (!done_mincut_) {
    FindMinCut();
  }
  return mincut_[node_id];
}

template <typename ValueType>
typename MaxflowGraph<ValueType>::value_type MaxflowGraph<ValueType>::GetCutValue(const std::vector<TermType>& cut) {
  value_type out = state_.st_offset;

  auto accumulate_caps = [&](std::size_t node_id) {
    for (auto arc_it = AliveArcBegin(node_id); arc_it != AliveArcEnd(node_id); ++arc_it) {
      auto arc = *arc_it;
      auto head_id = arc->GetHeadNode()->index;
      if (cut[head_id] == SINK) {
        out += GetArcBaseCap(arc);
      }
    }
  };

  accumulate_caps(GetSourceNode()->index);

  for (auto node_it = InnerBegin(); node_it != InnerEnd(); ++node_it) {
    auto node = *node_it;
    if (cut[node->index] == SOURCE) {
      accumulate_caps(node->index);
    }
  }

  return out;
}

template <typename ValueType>
std::vector<std::size_t> MaxflowGraph<ValueType>::GetMinCut(bool filter_variable) {
  if (!done_mincut_) {
    FindMinCut();
  }
  std::vector<std::size_t> indices;
  for (auto it = InnerBegin(); it != InnerEnd(); ++it) {
    auto node = *it;
    if ((!filter_variable || node->is_variable) && mincut_[node->index] == SOURCE) {
      indices.push_back(node->index);
    }
  }
  return indices;
}

template <typename ValueType>
typename MaxflowGraph<ValueType>::value_type
MaxflowGraph<ValueType>::GetCutValueByIds(const std::vector<std::size_t>& node_ids) {
  std::vector<TermType> cut(nodes_.size(), SINK);
  cut[GetSourceNode()->index] = SOURCE;
  for (const auto& node_id: node_ids) {
    cut[node_id] = SOURCE;
  }
  return GetCutValue(cut);
}

template <typename ValueType>
typename MaxflowGraph<ValueType>::value_type
MaxflowGraph<ValueType>::GetCutValueByNames(const std::vector<element_type>& members) {
  std::vector<TermType> cut(nodes_.size(), SINK);
  cut[GetSourceNode()->index] = SOURCE;
  for (const auto& name: members) {
    cut[name2id_[name]] = SOURCE;
  }
  return GetCutValue(cut);
}

template <typename ValueType>
void MaxflowGraph<ValueType>::_RearrangeAliveArcs(std::size_t node_id) {
  std::size_t first_alive = state_.first_alive_arc_index[node_id];
  for (auto i = state_.first_alive_arc_index[node_id]; i < adj_[node_id].size(); ++i) {
    auto arc = adj_[node_id][i];
    if (!IsArcAlive(arc)) {
      if (i != first_alive) {
        std::swap(adj_[node_id][i], adj_[node_id][first_alive]);
      }
      first_alive++;
    }
  }
  state_.first_alive_arc_index[node_id] = first_alive;
}

template <typename ValueType>
void MaxflowGraph<ValueType>::RearrangeAliveArcs() {
  _RearrangeAliveArcs(GetSourceNode()->index);
  _RearrangeAliveArcs(GetSinkNode()->index);
  for (auto it = InnerBegin(); it != InnerEnd(); ++it){
    _RearrangeAliveArcs((*it)->index);
  }
}

template <typename ValueType>
void MaxflowGraph<ValueType>::MakeSource(const Node_s& node) {

  std::size_t index = node->index;

  for (auto it = AliveArcBegin(index); it != AliveArcEnd(index); ++it) {
    auto arc = *it;
    value_type cap = GetArcBaseCap(arc);
    auto head = arc->GetHeadNode();
    if (IsInnerNode(head)) {
      head->sv_cap += cap;
    }
    if (IsSinkNode(head)) {
      state_.st_offset += cap;
    }
  }

  auto old_typed_index = node->typed_index;
  if (old_typed_index != state_.first_inner_index) {
    std::swap(typed_node_list_[state_.first_inner_index], typed_node_list_[old_typed_index]);
    node->typed_index = state_.first_inner_index;
    typed_node_list_[old_typed_index]->typed_index = old_typed_index;
  }
  state_.first_inner_index++;
}

template <typename ValueType>
void MaxflowGraph<ValueType>::UnmakeSource(const Node_s& node) {

  for (const auto &arc : adj_[node->index]) {
    value_type cap = GetArcBaseCap(arc);
    auto head = arc->GetHeadNode();
    if (IsArcAlive(arc)) {
      if (IsInnerNode(head)) {
        head->sv_cap -= cap;
      }
    }
    if (IsSinkNode(head)) {
      state_.st_offset -= cap;
    }
  }
}

template <typename ValueType>
void MaxflowGraph<ValueType>::MakeSink(const Node_s& node) {
  std::size_t index = node->index;

  for (auto it = AliveArcBegin(index); it != AliveArcEnd(index); ++it) {
    auto reversed = (*it)->GetReversed();
    value_type cap = GetArcBaseCap(reversed);
    auto tail = reversed->GetTailNode();
    if (IsInnerNode(tail)) {
      tail->vt_cap += cap;
    }
    if (IsSourceNode(tail)) {
      state_.st_offset += cap;
    }
  }

  auto old_typed_index = node->typed_index;
  if (old_typed_index != state_.first_sink_index - 1) {
    std::swap(typed_node_list_[state_.first_sink_index - 1], typed_node_list_[old_typed_index]);
    node->typed_index = state_.first_sink_index - 1;
    typed_node_list_[old_typed_index]->typed_index = old_typed_index;
  }
  state_.first_sink_index--;
}

template <typename ValueType>
void MaxflowGraph<ValueType>::UnmakeSink(const Node_s& node) {

  for (const auto &arc : adj_[node->index]) {
    auto reversed = arc->GetReversed();
    auto tail = reversed->GetTailNode();
    value_type cap = GetArcBaseCap(reversed);
    if (IsArcAlive(reversed)) {
      if (IsInnerNode(tail)) {
        tail->vt_cap -= cap;
      }
    }
    if (IsSourceNode(tail)) {
      state_.st_offset += cap;
    }
  }
}

template <typename ValueType>
MaxflowState<ValueType> MaxflowGraph<ValueType>::GetState() const {
  return state_;
}

template <typename ValueType>
void MaxflowGraph<ValueType>::RestoreState(MaxflowState<ValueType> state) {
  while (state_.first_inner_index > state.first_inner_index) {
    state_.first_inner_index--;
    auto node = typed_node_list_[state_.first_inner_index];
    UnmakeSource(node);
  }

  while (state_.first_sink_index < state.first_sink_index) {
    state_.first_sink_index++;
    auto node = typed_node_list_[state_.first_sink_index - 1];
    UnmakeSink(node);
  }

  state_ = state;
  done_max_preflow_ = false;
  done_mincut_ = false;
}

template <typename ValueType>
void MaxflowGraph<ValueType>::Reduction(const std::vector<TermType>& cut) {
  for (const auto& node: nodes_) {
    if (IsInnerNode(node) && cut[node->index] == SINK) {
      MakeSink(node);
    }
  }
  RearrangeAliveArcs();
  done_max_preflow_ = false;
  done_mincut_ = false;
}

template <typename ValueType>
void MaxflowGraph<ValueType>::ReductionByIds(const std::vector<std::size_t>& node_ids) {
  std::vector<TermType> cut(nodes_.size(), SINK);
  cut[GetSourceNode()->index] = SOURCE;
  for (const auto& node_id: node_ids) {
    cut[node_id] = SOURCE;
  }
  Reduction(cut);
}

template <typename ValueType>
void MaxflowGraph<ValueType>::ReductionByNames(const std::vector<element_type>& members) {
  std::vector<TermType> cut(nodes_.size(), SINK);
  cut[GetSourceNode()->index] = SOURCE;
  for (const auto& name: members) {
    cut[name2id_[name]] = SOURCE;
  }
  Reduction(cut);
}

template <typename ValueType>
void MaxflowGraph<ValueType>::Contraction(const std::vector<TermType>& cut, value_type additional_offset) {
  for (const auto& node: nodes_) {
    if (IsInnerNode(node) && cut[node->index] == SOURCE) {
      MakeSource(node);
    }
  }
  RearrangeAliveArcs();
  state_.st_offset += additional_offset;
  done_max_preflow_ = false;
  done_mincut_ = false;
}

template <typename ValueType>
void MaxflowGraph<ValueType>::ContractionByIds(const std::vector<std::size_t>& node_ids,
                                    value_type additional_offset)
{
  std::vector<TermType> cut(nodes_.size(), SINK);
  cut[GetSourceNode()->index] = SOURCE;
  for (const auto& node_id: node_ids) {
    cut[node_id] = SOURCE;
  }
  Contraction(cut, additional_offset);
}

template <typename ValueType>
void MaxflowGraph<ValueType>::ContractionByNames(const std::vector<element_type>& members,
                                    value_type additional_offset)
{
  std::vector<TermType> cut(nodes_.size(), SINK);
  cut[GetSourceNode()->index] = SOURCE;
  for (const auto& name: members) {
    cut[name2id_[name]] = SOURCE;
  }
  Contraction(cut, additional_offset);
}

template <typename ValueType>
std::vector<std::size_t> MaxflowGraph<ValueType>::GetInnerIndices(bool filter_variable) const {
  std::vector<std::size_t> indices;
  for (auto it = InnerBegin(); it != InnerEnd(); ++it) {
    if (!filter_variable || (*it)->is_variable) {
      indices.push_back((*it)->index);
    }
  }
  return indices;
}

template <typename ValueType>
std::vector<element_type> MaxflowGraph<ValueType>::GetMembers() const {
  std::vector<element_type> members;
  for (auto it = InnerBegin(); it != InnerEnd(); ++it) {
    if ((*it)->is_variable) {
      auto node_id = (*it)->index;
      members.push_back(nodes_[node_id]->name);
    }
  }
  return members;
}

}

#endif
