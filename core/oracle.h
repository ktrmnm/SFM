#ifndef ORACLE_H
#define ORACLE_H

#include <vector>
#include <utility>
#include <type_traits>
#include "core/utils.h"
#include "core/set_utils.h"
#include "partial_vector.h"

namespace submodular {

template <typename T>
struct ValueTraits{
  using value_type = typename std::enable_if<std::is_arithmetic<T>::value, T>::type;
  using rational_type = typename std::conditional<std::is_floating_point<T>::value, T, double>::type;
  using base_type = PartialVector<rational_type>;
};

template <typename ValueType>
class SubmodularOracle {
public:
  using value_type = typename ValueTraits<ValueType>::value_type;
  using rational_type = typename ValueTraits<ValueType>::rational_type;

  virtual std::size_t GetN() = 0;
  virtual std::size_t GetNGround() = 0;
  virtual value_type Call(const Set& X) = 0;
  //value_type operator()(const Set& X) { return Call(X); }

  Set GetDomain() const { return domain_.Copy(); }
  void SetDomain(const Set& X) { domain_ = X; }
  void SetDomain(Set&& X) { domain_ = std::move(X); }
protected:
  Set domain_;
};

/*
template <typename OracleType>
struct OracleTraits {
  using value_type = typename OracleType::value_type;
  using rational_type = typename OracleType::rational_type;
  using base_type = PartialVector<rational_type>;
};
*/

}

#endif
