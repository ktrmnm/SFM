#ifndef ORACLES_MODULAR_H
#define ORACLES_MODULAR_H

#include <vector>
#include <stdexcept>
#include "core/oracle.h"
#include "core/set_utils.h"
#include "core/utils.h"

namespace submodular {

template <typename ValueType>
class ModularOracle: public SubmodularOracle<ValueType> {
public:
  using value_type = typename utils::value_traits<ValueType>::value_type;
  using rational_type = typename utils::value_traits<ValueType>::rational_type;

  ModularOracle(const std::vector<value_type>& x)
    :n_(x.size()), x_(x) { this->SetDomain(Set::MakeDense(n_)); }
  ModularOracle(std::vector<value_type>&& x)
    :n_(x.size()), x_(std::move(x)) { this->SetDomain(Set::MakeDense(n_)); }

  value_type Call(const Set& X) {
    if (X.n_ != n_) {
      throw std::range_error("ModularOracle::Call: Input size mismatch");
    }
    value_type f(0);
    for (auto& i: X.GetMembers()) {
      f += x_[i];
    }
    return f;
  }

  std::size_t GetN() { return n_; }
  std::size_t GetNGround() { return n_; }

private:
  std::size_t n_;
  std::vector<value_type> x_;
};

template <typename ValueType>
class ConstantOracle: public SubmodularOracle<ValueType> {
public:
  using value_type = typename utils::value_traits<ValueType>::value_type;
  using rational_type = typename utils::value_traits<ValueType>::rational_type;

  ConstantOracle(std::size_t n, value_type c): n_(n), c_(c) { this->SetDomain(Set::MakeDense(n_)); }

  value_type Call(const Set& X) {
    if (X.n_ != n_) {
      throw std::range_error("ConstantOracle::Call: Input size mismatch");
    }
    return c_;
  }

  std::size_t GetN() { return n_; }
  std::size_t GetNGround() { return n_; }

private:
  std::size_t n_;
  value_type c_;
};


}

#endif
