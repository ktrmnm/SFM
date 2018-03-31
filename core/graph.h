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

#ifndef GRAPH_H
#define GRAPH_H

#include <iterator>
#include <memory>
#include <utility>
#include <vector>
#include <list>
#include <unordered_map>
#include <deque>
#include <algorithm>
#include "core/utils.h"
#include "core/set_utils.h"


namespace submodular {

using Height_t = int;
enum TermType { SOURCE = 1, SINK = 0, INNER = 2 };
enum NodeColor { WHITE, GRAY, BLACK };

template <typename NodeType>
struct NodeTraits {
  using type_s = std::shared_ptr<NodeType>;
  using type_w = std::weak_ptr<NodeType>;
  using arc_type = typename NodeType::arc_type;
  using value_type = typename NodeType::value_type;
  using color_type = typename NodeType::color_type;
};

template <typename ArcType>
struct ArcTraits {
  using type_s = std::shared_ptr<ArcType>;
  using type_w = std::weak_ptr<ArcType>;
  using node_type = typename ArcType::node_type;
  using value_type = typename ArcType::value_type;
};

template <typename GraphType>
struct GraphTraits {
  using node_type = typename GraphType::node_type;
  using arc_type = typename GraphType::arc_type;
  using Node_s = typename NodeTraits<node_type>::type_s;
  using Arc_s = typename ArcTraits<arc_type>::type_s;
  using value_type = typename GraphType::value_type;
  using state_type = typename GraphType::state_type;
  using node_iterator = typename GraphType::node_iterator;
  using c_node_iterator = typename GraphType::c_node_iterator;
  using arc_iterator = typename GraphType::arc_iterator;
  using c_arc_iterator = typename GraphType::c_arc_iterator;
};

template <typename ValueType> struct Node;
template <typename ValueType> struct Arc;
template <typename ValueType> class SimpleGraph;

template <typename ValueType>
struct Node {
  using arc_type = std::nullptr_t;
  using value_type = ValueType;
  using color_type = NodeColor;
  element_type name;
  //std::size_t index;
  value_type excess;
  NodeColor color;
};

template <typename ValueType>
struct Arc {
  using node_type = Node<ValueType>;
  using value_type = ValueType;
  using Node_s = typename NodeTraits<node_type>::type_s;
  using Node_w = typename NodeTraits<node_type>::type_w;
  using Arc_s  = typename ArcTraits<Arc<ValueType>>::type_s;
  using Arc_w  = typename ArcTraits<Arc<ValueType>>::type_w;

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
class SimpleGraph {
public:
  using node_type = Node<ValueType>;
  using arc_type = Arc<ValueType>;
  using value_type = ValueType;
  using state_type = std::nullptr_t;
  using Node_s = typename NodeTraits<node_type>::type_s;
  using Node_w = typename NodeTraits<node_type>::type_w;
  using Arc_s = typename ArcTraits<arc_type>::type_s;
  using Arc_w = typename ArcTraits<arc_type>::type_w;

  using node_iterator = typename utils::unordered_map_value_iterator<element_type, Node_s>;
  using c_node_iterator = const typename utils::unordered_map_value_iterator<element_type, Node_s>;
  using arc_iterator = typename utils::unordered_map_value_iterator<element_type, Arc_s>;
  using c_arc_iterator = const typename utils::unordered_map_value_iterator<element_type, Arc_s>;

private:
  std::unordered_map<element_type, Node_s> nodes_;
  //std::vector<Arc_s> arcs_;
  std::unordered_map<element_type, std::unordered_map<element_type, Arc_s>> adj_;
  value_type tol_;

  //std::size_t arc_count_; // used for hash of arc
  Arc_s MakeArc(const Node_s &head, const Node_s &tail, value_type cap);

public:
  SimpleGraph(): tol_(1e-8) {};
  SimpleGraph(const SimpleGraph&) = default;
  SimpleGraph(SimpleGraph&&) = default;
  SimpleGraph& operator=(const SimpleGraph&) = default;
  SimpleGraph& operator=(SimpleGraph&&) = default;

  void SetTol(value_type tol) { tol_ = tol; }

  // Methods for graph construction
  //void Reserve(std::size_t n, std::size_t m);
  Node_s AddNode(element_type name);
  void RemoveNode(element_type name);
  void AddArc(const Node_s &head, const Node_s &tail, value_type cap);
  void AddArcPair(const Node_s &head, const Node_s &tail, value_type cap, value_type rev_cap);
  //void MakeGraph();

  // Methods to get graph information
  std::size_t GetNodeNumber() const { return nodes_.size(); }
  bool HasNode(element_type name) const { return nodes_.count(name) == 1; }
  Node_s GetNode(element_type name) const {
    return HasNode(name) ? nodes_.at(name) : nullptr;
  }
  Arc_s GetArc(element_type head_name, element_type tail_name);
  Arc_s GetArc(const Node_s& head, const Node_s& tail);
  bool IsArcAlive(const Arc_s& arc) const {
    return utils::is_abs_close(arc->flow, value_type(0), tol_);
    //return arc->flow <= value_type(0);
  }

  void ClearFlow();
  //void ClearExcess();
  void Push(const Arc_s& arc, value_type amount);
  void Augment(const Arc_s& arc, value_type amount);

  class node_range {
  public:
    using orig_iterator = typename std::unordered_map<element_type, Node_s>::iterator;
  private:
    node_iterator begin_;
    node_iterator end_;
  public:
    node_range() = delete;
    node_range(orig_iterator begin, orig_iterator end): begin_(begin), end_(end) {}
    node_iterator& begin() noexcept { return this->begin_; }
    const node_iterator& begin() const noexcept { return this->begin_; }
    node_iterator& end() noexcept { return this->end_; }
    const node_iterator& end() const noexcept { return this->end_; }
  };

  node_range NodeRange() noexcept { return node_range(nodes_.begin(), nodes_.end()); }
  //const node_range CNodeRange() noexcept { return node_range(nodes_.begin(), nodes_.end()); }

  class arc_range {
  public:
    using orig_iterator = typename std::unordered_map<element_type, Arc_s>::iterator;
  private:
    arc_iterator begin_;
    arc_iterator end_;
  public:
    arc_range() = delete;
    arc_range(orig_iterator begin, orig_iterator end): begin_(begin), end_(end) {}
    arc_iterator& begin() noexcept { return this->begin_; }
    const arc_iterator& begin() const noexcept { return this->begin_; }
    arc_iterator& end() noexcept { return this->end_; }
    const arc_iterator& end() const noexcept { return this->end_; }
  };

  arc_range OutArcRange(element_type name) noexcept { return arc_range(adj_.at(name).begin(), adj_.at(name).end()); }
  arc_range OutArcRange(const Node_s& node) noexcept { return OutArcRange(node->name); }
};

template <typename ValueType>
typename SimpleGraph<ValueType>::Node_s SimpleGraph<ValueType>::AddNode(element_type name) {
  if (!HasNode(name)) {
    auto node = std::make_shared<node_type>();
    node->name = name;
    node->excess = 0;
    nodes_[name] = node;
    return node;
  }
  else {
    return nodes_[name];
  }
}

template <typename ValueType>
typename SimpleGraph<ValueType>::Arc_s
SimpleGraph<ValueType>::MakeArc(const Node_s &head, const Node_s &tail, value_type cap) {
  auto arc = std::make_shared<arc_type>();
  arc->flow = 0;
  arc->capacity = cap;
  arc->head_node = head;
  arc->tail_node = tail;
  //arcs_.push_back(arc);
  arc->hash = head->name;
  //arc_count_++;
  return arc;
}

template <typename ValueType>
void SimpleGraph<ValueType>::AddArc(const Node_s &head, const Node_s &tail, value_type cap) {
  auto arc = MakeArc(head, tail, cap);
  adj_[tail->name][arc->name] = arc;
}

template <typename ValueType>
void SimpleGraph<ValueType>::AddArcPair(const Node_s &head, const Node_s &tail, value_type cap, value_type rev_cap) {
  auto arc = MakeArc(head, tail, cap);
  auto rev = MakeArc(tail, head, rev_cap);
  arc->reversed = rev;
  rev->reversed = arc;
  adj_[tail->name][arc->hash] = arc;
  adj_[head->name][rev->hash] = rev;
}

template <typename ValueType>
typename SimpleGraph<ValueType>::Arc_s
SimpleGraph<ValueType>::GetArc(element_type head_name, element_type tail_name) {
  if (HasNode(head_name) && HasNode(tail_name) && adj_[tail_name].count(head_name) == 1) {
    return adj_[tail_name][head_name];
  }
  else {
    return nullptr;
  }
}

template <typename ValueType>
typename SimpleGraph<ValueType>::Arc_s
SimpleGraph<ValueType>::GetArc(const Node_s& head, const Node_s& tail) {
  return GetArc(head->name, tail->name);
}

template <typename ValueType>
void SimpleGraph<ValueType>::RemoveNode(element_type name) {
  if (HasNode(name)) {
    for (auto&& arc: OutArcRange(name)) {
      auto dst = arc->GetHeadNode()->name;
      auto rev_hash = arc->GetReversed()->hash;
      adj_[dst].erase(rev_hash);
    }
    adj_[name].clear();
    nodes_.erase(name);
  }
}

template <typename ValueType>
void SimpleGraph<ValueType>::ClearFlow() {
  for (auto&& node: NodeRange()) {
    node->excess = 0;
    for (auto&& arc: OutArcRange(node->name)) {
      arc->flow = 0;
    }
  }
}

template <typename ValueType>
void SimpleGraph<ValueType>::Push(const Arc_s& arc, value_type amount) {
  auto rev = arc->GetReversed();
  auto head = arc->GetHeadNode();
  auto tail = arc->GetTailNode();
  arc->flow += amount;
  rev->flow -= amount;
  tail->excess -= amount;
  head->excess += amount;
}

template <typename ValueType>
void SimpleGraph<ValueType>::Augment(const Arc_s& arc, value_type amount) {
  auto rev = arc->GetReversed();
  auto head = arc->GetHeadNode();
  auto tail = arc->GetTailNode();
  auto delta = std::max(amount - rev->flow, value_type(0));
  auto delta_rev = std::min(amount, rev->flow);
  arc->flow += delta;
  rev->flow -= delta_rev;
  tail->excess -= amount;
  head->excess += amount;
}

template <typename ValueType>
SimpleGraph<ValueType> MakeCompleteGraph(const Set& nodes, ValueType cap = 0) {
  SimpleGraph<ValueType> G;
  auto node_ids = nodes.GetMembers();
  for (const auto& name: node_ids) {
    G.AddNode(name);
  }
  for (element_type i = 0; i < node_ids.size(); ++i) {
    for (element_type j = 0; j < i; ++j) {
      auto src = G.GetNode(i);
      auto dst = G.GetNode(j);
      G.AddArcPair(dst, src, cap, cap);
    }
  }
  return G;
}

template <typename GraphType>
std::vector<element_type> GetReachableIndicesFrom(GraphType& G, element_type start_id) {
  std::vector<element_type> reachable;

  auto start_node = G.GetNode(start_id);

  if (start_node != nullptr) {
    for (auto&& node: G.NodeRange()) {
      node->color = WHITE;
    }
    start_node->color = BLACK;
    reachable.push_back(start_node->name);
    std::deque<typename GraphTraits<GraphType>::Node_s> Q;
    Q.push_back(start_node);

    while (!Q.empty()) {//BFS start
      auto node = Q.front();
      Q.pop_front();
      for (auto&& arc: G.OutArcRange(node)) {
        if (G.IsArcAlive(arc)) {
          auto dst = arc->GetHeadNode();
          if (dst->color == WHITE) {
            dst->color = BLACK;
            reachable.push_back(dst->name);
            Q.push_back(dst);
          }
        }
      }
    }//BFS end
  }
  return reachable;
}

template <typename GraphType>
std::vector<element_type> GetReachableIndicesFrom(GraphType& G, std::vector<element_type> start_ids) {
  std::vector<element_type> reachable;

  for (auto&& node: G.NodeRange()) {
    node->color = WHITE;
  }

  for (const auto& start_id: start_ids) {//for start_ids
    auto start_node = G.GetNode(start_id);
    if (start_node != nullptr && start_node->color == WHITE) {
      start_node->color = BLACK;
      reachable.push_back(start_node->name);
      std::deque<typename GraphTraits<GraphType>::Node_s> Q;
      Q.push_back(start_node);

      while (!Q.empty()) {//BFS start
        auto node = Q.front();
        Q.pop_front();
        for (auto&& arc: G.OutArcRange(node)) {
          if (G.IsArcAlive(arc)) {
            auto dst = arc->GetHeadNode();
            if (dst->color == WHITE) {
              dst->color = BLACK;
              reachable.push_back(dst->name);
              Q.push_back(dst);
            }
          }
        }
      }//BFS end
    }
  }//for start_ids

  return reachable;
}

template <typename GraphType>
auto FindSTPath(GraphType& G, const Set& S, const Set& T) {
  std::list<typename GraphTraits<GraphType>::Arc_s> path;
  std::unordered_map<element_type, typename GraphTraits<GraphType>::Arc_s> parent_arc;

  for (auto&& node: G.NodeRange()) {
    node->color = WHITE;
  }

  auto start_ids = S.GetMembers();
  for (const auto& start_id: start_ids) {//for start_ids
    auto start_node = G.GetNode(start_id);
    path.clear();
    parent_arc.clear();
    if (start_node != nullptr && start_node->color == WHITE) {
      start_node->color = BLACK;
      parent_arc[start_node->name] = nullptr;
      std::vector<typename GraphTraits<GraphType>::Node_s> stack;
      stack.push_back(start_node);

      while (!stack.empty()) {//DFS
        auto node = stack.back();
        stack.pop_back();
        for (auto&& arc: G.OutArcRange(node)) {
          if (G.IsArcAlive(arc)) {
            auto dst = arc->GetHeadNode();

            if (dst->color == WHITE) {
              dst->color = BLACK;
              parent_arc[dst->name] = arc;
              stack.push_back(dst);
              if (T.HasElement(dst->name)) {// make path
                auto parent = parent_arc[dst->name];
                while (parent != nullptr) {
                  path.push_front(parent);
                  parent = parent_arc[parent->GetTailNode()->name];
                }
                return path;
              }// make path
            }
          }
        }
      }//DFS
    }
  }//for start_ids
  path.clear();
  return path;
}

}

#endif
