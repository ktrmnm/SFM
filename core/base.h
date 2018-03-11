#ifndef BASE_H
#define BASE_H

#include <vector>
#include <utility>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include "core/set_utils.h"
#include "core/oracle.h"

namespace submodular {

using OrderType = std::vector<std::size_t>;

template <typename T>
class IndexCompare {
public:
  std::vector<T> x_;
  IndexCompare() = delete;
  explicit IndexCompare(const std::vector<T>& x): x_(x) {}
  bool operator()(std::size_t i, std::size_t j) {
    return x_.at(i) > x_.at(j);
  }
};

template <typename T>
OrderType GetDescendingOrder(const std::vector<T>& x) {
  IndexCompare<T> comparer(x);
  OrderType order(x.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), comparer);
  return order;
}

template <typename OracleType>
auto GreedyBase(OracleType& F, const OrderType& order) {
  auto n = F.GetN();
  auto n_ground = F.GetNGround();
  if (order.size() != n) {
    throw std::range_error("GreedyBase:: Domain size mismatch");
  }
  auto base_domain = F.GetDomain();
  typename OracleTraits<OracleType>::base_type base(base_domain);

  auto X = Set::MakeEmpty(n_ground);
  auto prev = F.Call(X);
  auto curr = prev;
  for (const auto& i: order) {
    X.AddElement(i);
    curr = F.Call(X);
    base[i] = curr - prev;
    prev = curr;
  }
  return base;
}

}

#endif
