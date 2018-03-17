#ifndef GRAPH_H
#define GRAPH_H

#include <iterator>
#include <memory>
#include <utility>
#include <vector>
#include <unordered_map>
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
  using value_type = typename GraphType::value_type;
  using state_type = typename GraphType::state_type;
  using node_iterator = typename GraphType::node_iterator;
  using c_node_iterator = typename GraphType::c_node_iterator;
  using arc_iterator = typename GraphType::arc_iterator;
  using c_arc_iterator = typename GraphType::c_arc_iterator;
}

template <typename ValueType> struct Node;
template <typename ValueType> struct Arc;
template <typename ValueType> struct SimpleGraph;

template <typename ValueType>
struct Node {
  using arc_type = nullptr_t;
  using value_type = ValueType;
  std::size_t index;
  value_type excess;
  //std::size_t GetIndex() const { return index; }
};

template <typename ValueType>
struct Arc {
  using node_type = Node<ValueType>;
  using value_type = ValueType;
  using Node_s = typename NodeTraits<node_type>::type_s;
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

  using node_iterator = typename unordered_map_value_iterator<std::size_t, Node_s>;
  using c_node_iterator = typename const unordered_map_value_iterator<std::size_t, Node_s>;
  using arc_iterator = typename unordered_map_value_iterator<std::size_t, Arc_s>;
  using c_arc_iterator = typename const unordered_map_value_iterator<std::size_t, Arc_s>;

private:
  std::unordered_map<Node_s> nodes_;
  //std::vector<Arc_s> arcs_;
  std::unordered_map<std::unordered_map<Arc_s>> adj_;
  value_type tol_;

  std::size_t arc_count_; // used for hash of arc
  Arc_s MakeArc(const Node_s &head, const Node_s &tail, value_type cap);

public:
  SimpleGraph(): arc_count_(0), tol_(1e-8) {};
  SimpleGraph(const Graph&) = default;
  SimpleGraph(Graph&&) = default;
  SimpleGraph& operator=(const Graph&) = default;
  SimpleGraph& operator=(Graph&&) = default;

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
    return HasNode(index) ? nodes_[index] : nullptr;
  }
  bool IsArcAlive(const Arc_s& arc) const {
    return utils::is_abs_close(arc->flow, value_type(0), tol_);
  }

  void ClearFlow();
  //void ClearExcess();
  void Push(const Arc_s& arc, value_type amount);

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

  class arc_range {
  public:
    using orig_iterator = typename std::unordered_map<std::size_t, Arc_s>::iterator;
  private:
    arc_iterator begin_;
    arc_iterator end_;
  public:
    arc_range() = delete;
    arc_range(iterator begin, iterator end): begin_(begin), end_(end) {}
    arc_iterator& begin() noexcept { return this->begin_; }
    const arc_iterator& begin() const noexcept { return this->begin_; }
    arc_iterator& end() noexcept { return this->end_; }
    const arc_iterator& end() const noexcept { return this->end_; }
  };

  arc_range OutArcRange(std::size_t index) { return arc_range(adj_[index].begin(), adj_[index].end()); }
  arc_range OutArcRange(const Node_s& node) { return OutArcRange(node->index); }
};

template <typename ValueType>
typename SimpleGraph<ValueType>::Node_s SimpleGraph<ValueType>::AddNode(std::size_t index) {
  if (!HasNode(index)) {
    auto node = std::make_shared<node_type>();
    node->index = index;
    node->excess = 0;
    nodes_[index] = node;
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
  arc->hash = arc_count_;
  arc_count_++;
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
void SimpleGraph<ValueType>::RemoveNode(std::size_t index) {
  if (HasNode(index)) {
    for (const auto& arc: OutArcRange(index)) {
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
  for (const auto& node: NodeRange()) {
    node->excess = 0;
    for (const auto& arc: OutArcRange(node->index)) {
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

}

#endif
