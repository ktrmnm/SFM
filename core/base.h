#ifndef BASE_H
#define BASE_H

#include <vector>
#include <utility>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include "core/set_utils.h"
#include "core/oracle.h"
#include "core/linalg.h"

namespace submodular {

using OrderType = std::vector<std::size_t>;

OrderType LinearOrder(std::size_t n) {
  OrderType order(n);
  std::iota(order.begin(), order.end(), 0);
  return order;
}

OrderType LinearOrder(const Set& X) {
  return X.GetMembers();
}

template <typename T>
OrderType GetDescendingOrder(const std::vector<T>& x) {
  OrderType order(x.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](std::size_t i, std::size_t j){ return x[i] > x[j]; });
  return order;
}

template <typename T>
OrderType GetDescendingOrder(const PartialVector<T>& x) {
  OrderType order = LinearOrder(x.GetDomain());
  std::sort(order.begin(), order.end(), [&](std::size_t i, std::size_t j){ return x[i] > x[j]; });
  return order;
}

template <typename ValueType>
auto GreedyBase(SubmodularOracle<ValueType>& F, const OrderType& order) {
  auto n = F.GetN();
  auto n_ground = F.GetNGround();
  if (order.size() != n) {
    throw std::range_error("GreedyBase:: Domain size mismatch");
  }
  auto base_domain = F.GetDomain();
  typename ValueTraits<ValueType>::base_type base(base_domain);

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

template <typename ValueType>
class BaseCombination {
public:
  using rational_type = typename ValueTraits<ValueType>::rational_type;
  using base_type = typename ValueTraits<ValueType>::base_type;

  BaseCombination() = default;
  BaseCombination(const BaseCombination&) = default;
  BaseCombination(BaseCombination&&) = default;
  BaseCombination& operator=(const BaseCombination&) = default;
  BaseCombination& operator=(BaseCombination&&) = default;
  explicit BaseCombination(const Set& domain): n_ground_(domain.n_), domain_(domain) {}
  explicit BaseCombination(Set&& domain): n_ground_(domain.n_), domain_(std::move(domain)) {}

  std::size_t Size() const { return orders_.size(); }
  base_type GetBase(std::size_t i) const { return bases_[i]; }
  base_type& GetBase(std::size_t i) { return bases_[i]; }
  OrderType GetOrder(std::size_t i) const { return orders_[i]; }
  OrderType& GetOrder(std::size_t i) { return orders_[i]; }
  rational_type GetCoeff(std::size_t i) const { return coeffs_[i]; }
  rational_type& GetCoeff(std::size_t i) { return coeffs_[i]; }
  //void SetCoeff(std::size_t i, rational_type coeff);
  base_type GetCombination() const;

  void AddTriple(const OrderType& order, const base_type& base, rational_type coeff);
  void AddTriple(OrderType&& order, base_type&& base, rational_type coeff);

  bool IsActive(std::size_t order_id, const Set& X) const;
  std::pair<std::size_t, std::size_t> GetActivePair(std::size_t order_id, const Set& X) const;

  void Reduce();

  //bool IsConvexCombination(rational_type tol);

  std::size_t n_ground_;
  Set domain_;
  std::vector<OrderType> orders_;
  std::vector<base_type> bases_;
  std::vector<rational_type> coeffs_;
};

template <typename ValueType>
typename BaseCombination<ValueType>::base_type
BaseCombination<ValueType>::GetCombination() const {
  base_type x(domain_); //initialized by all zero partial vector
  for (std::size_t i = 0; i < Size(); ++i) {
    x += coeffs_[i] * bases_[i];
  }
  return x;
}

template <typename ValueType>
void BaseCombination<ValueType>::AddTriple(const OrderType& order, const base_type& base, rational_type coeff) {
  orders_.push_back(order);
  bases_.push_back(base);
  coeffs_.push_back(coeff);
}

template <typename ValueType>
void BaseCombination<ValueType>::AddTriple(OrderType&& order, base_type&& base, rational_type coeff) {
  orders_.push_back(std::move(order));
  bases_.push_back(std::move(base));
  coeffs_.push_back(coeff);
}

template <typename ValueType>
bool BaseCombination<ValueType>::IsActive(std::size_t order_id, const Set& X) const {
  auto order = orders_[order_id];
  if (order.size() >= 2) {
    for (std::size_t i = 0; i < order.size() - 1; ++i) {
      if (!X[order[i]] && X[order[i + 1]]) {
        return true;
      }
    }
  }
  return false;
}

template <typename ValueType>
std::pair<std::size_t, std::size_t>
BaseCombination<ValueType>::GetActivePair(std::size_t order_id, const Set& X) const {
  std::size_t u, v;
  auto order = orders_[order_id];
  for (std::size_t i = 0; i < order.size() - 1; ++i) {
    if (!X[order[i]] && X[order[i + 1]]) {
      v = order[i];
      u = order[i + 1];
    }
  }
  return std::make_pair(u, v);
}

template <typename ValueType>
void BaseCombination<ValueType>::Reduce() {
  std::size_t n = domain_.Cardinality();
  std::vector<std::vector<double>> Y;
  Y.reserve(bases_.size());
  for (std::size_t i = 0; i < bases_.size(); ++i) {
    auto data = bases_[i].GetActiveVector();
    std::vector<double> data_d(data.begin(), data.end());
    Y.push_back(std::move(data_d));
  }
  std::vector<double> C(coeffs_.begin(), coeffs_.end());
  std::vector<OrderType> orders(orders_);
  auto m = linalg::reduce_bases_with_order_swap(n, Y, C, orders);

  bases_.clear();
  coeffs_.clear();
  orders_.clear();
  for (std::size_t i = 0; i < m; ++i) {
    base_type base(domain_);
    std::vector<rational_type> data(Y[i].begin(), Y[i].end());
    base.SetActiveVector(data);
    AddTriple(std::move(orders[i]), std::move(base), C[i]);
  }
}

}

#endif
