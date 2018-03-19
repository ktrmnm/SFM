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
template <typename ValueType> struct SimpleGraph;

template <typename ValueType>
struct Node {
  using arc_type = nullptr_t;
  using value_type = ValueType;
  using color_type = NodeColor;
  std::size_t index;
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
  using state_type = nullptr_t;
  using Node_s = typename NodeTraits<node_type>::type_s;
  using Node_w = typename NodeTraits<node_type>::type_w;
  using Arc_s = typename ArcTraits<arc_type>::type_s;
  using Arc_w = typename ArcTraits<arc_type>::type_w;

  using node_iterator = typename utils::unordered_map_value_iterator<std::size_t, Node_s>;
  using c_node_iterator = const typename utils::unordered_map_value_iterator<std::size_t, Node_s>;
  using arc_iterator = typename utils::unordered_map_value_iterator<std::size_t, Arc_s>;
  using c_arc_iterator = const typename utils::unordered_map_value_iterator<std::size_t, Arc_s>;

private:
  std::unordered_map<std::size_t, Node_s> nodes_;
  //std::vector<Arc_s> arcs_;
  std::unordered_map<std::size_t, std::unordered_map<std::size_t, Arc_s>> adj_;
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
  Node_s AddNode(std::size_t index);
  void RemoveNode(std::size_t index);
  void AddArc(const Node_s &head, const Node_s &tail, value_type cap);
  void AddArcPair(const Node_s &head, const Node_s &tail, value_type cap, value_type rev_cap);
  //void MakeGraph();

  // Methods to get graph information
  std::size_t GetNodeNumber() const { return nodes_.size(); }
  bool HasNode(std::size_t index) const { return nodes_.count(index) == 1; }
  Node_s GetNode(std::size_t index) const {
    return HasNode(index) ? nodes_.at(index) : nullptr;
  }
  Arc_s GetArc(std::size_t head_id, std::size_t tail_id);
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
    using orig_iterator = typename std::unordered_map<std::size_t, Node_s>::iterator;
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
    using orig_iterator = typename std::unordered_map<std::size_t, Arc_s>::iterator;
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

  arc_range OutArcRange(std::size_t index) noexcept { return arc_range(adj_.at(index).begin(), adj_.at(index).end()); }
  //const arc_range COutArcRange(std::size_t index) noexcept { return arc_range(adj_[index].begin(), adj_[index].end()); }
  arc_range OutArcRange(const Node_s& node) noexcept { return OutArcRange(node->index); }
  //const arc_range COutArcRange(const Node_s& node) noexcept { return OutArcRange(node->index); }
};

template <typename ValueType>
typename SimpleGraph<ValueType>::Node_s SimpleGraph<ValueType>::AddNode(std::size_t index) {
  if (!HasNode(index)) {
    auto node = std::make_shared<node_type>();
    node->index = index;
    node->excess = 0;
    nodes_[index] = node;
    return node;
  }
  else {
    return nodes_[index];
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
  arc->hash = head->index;
  //arc_count_++;
  return arc;
}

template <typename ValueType>
void SimpleGraph<ValueType>::AddArc(const Node_s &head, const Node_s &tail, value_type cap) {
  auto arc = MakeArc(head, tail, cap);
  adj_[tail->index][arc->hash] = arc;
}

template <typename ValueType>
void SimpleGraph<ValueType>::AddArcPair(const Node_s &head, const Node_s &tail, value_type cap, value_type rev_cap) {
  auto arc = MakeArc(head, tail, cap);
  auto rev = MakeArc(tail, head, rev_cap);
  arc->reversed = rev;
  rev->reversed = arc;
  adj_[tail->index][arc->hash] = arc;
  adj_[head->index][rev->hash] = rev;
}

template <typename ValueType>
typename SimpleGraph<ValueType>::Arc_s
SimpleGraph<ValueType>::GetArc(std::size_t head_id, std::size_t tail_id) {
  if (HasNode(head_id) && HasNode(tail_id) && adj_[tail_id].count(head_id) == 1) {
    return adj_[tail_id][head_id];
  }
  else {
    return nullptr;
  }
}

template <typename ValueType>
typename SimpleGraph<ValueType>::Arc_s
SimpleGraph<ValueType>::GetArc(const Node_s& head, const Node_s& tail) {
  return GetArc(head->index, tail->index);
}

template <typename ValueType>
void SimpleGraph<ValueType>::RemoveNode(std::size_t index) {
  if (HasNode(index)) {
    for (auto&& arc: OutArcRange(index)) {
      auto dst = arc->GetHeadNode()->index;
      auto rev_hash = arc->GetReversed()->hash;
      adj_[dst].erase(rev_hash);
    }
    adj_[index].clear();
    nodes_.erase(index);
  }
}

template <typename ValueType>
void SimpleGraph<ValueType>::ClearFlow() {
  for (auto&& node: NodeRange()) {
    node->excess = 0;
    for (auto&& arc: OutArcRange(node->index)) {
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
  for (const auto& index: node_ids) {
    G.AddNode(index);
  }
  for (std::size_t i = 0; i < node_ids.size(); ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      auto src = G.GetNode(i);
      auto dst = G.GetNode(j);
      G.AddArcPair(dst, src, cap, cap);
    }
  }
  return G;
}

template <typename GraphType>
std::vector<std::size_t> GetReachableIndicesFrom(GraphType& G, std::size_t start_id) {
  std::vector<std::size_t> reachable;

  auto start_node = G.GetNode(start_id);

  if (start_node != nullptr) {
    for (auto&& node: G.NodeRange()) {
      node->color = WHITE;
    }
    start_node->color = BLACK;
    reachable.push_back(start_node->index);
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
            reachable.push_back(dst->index);
            Q.push_back(dst);
          }
        }
      }
    }//BFS end
  }
  return reachable;
}

template <typename GraphType>
std::vector<std::size_t> GetReachableIndicesFrom(GraphType& G, std::vector<std::size_t> start_ids) {
  std::vector<std::size_t> reachable;

  for (auto&& node: G.NodeRange()) {
    node->color = WHITE;
  }

  for (const auto& start_id: start_ids) {//for start_ids
    auto start_node = G.GetNode(start_id);
    if (start_node != nullptr && start_node->color == WHITE) {
      start_node->color = BLACK;
      reachable.push_back(start_node->index);
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
              reachable.push_back(dst->index);
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
  std::unordered_map<std::size_t, typename GraphTraits<GraphType>::Arc_s> parent_arc;

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
      parent_arc[start_node->index] = nullptr;
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
              parent_arc[dst->index] = arc;
              stack.push_back(dst);
              if (T.HasElement(dst->index)) {// make path
                auto parent = parent_arc[dst->index];
                while (parent != nullptr) {
                  path.push_front(parent);
                  parent = parent_arc[parent->GetTailNode()->index];
                }
                return path;
              }// make path
            }
          }
        }
      }//DFS
    }
  }//for start_ids

  return path;
}

}

#endif
