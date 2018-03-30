#ifndef ORACLES_IWATA_TEST_FUNCTION_H
#define ORACLES_IWATA_TEST_FUNCTION_H

#include <vector>
#include <stdexcept>
//#include <numeric>
#include "core/oracle.h"
#include "core/set_utils.h"

namespace submodular {

template <typename ValueType>
class IwataTestFunction: public SubmodularOracle<ValueType> {
public:
  using value_type = typename ValueTraits<ValueType>::value_type;

  IwataTestFunction(std::size_t n): n_(n) { this->SetDomain(Set::MakeDense(n_)); }

  value_type Call(const Set& X) {
    if (X.n_ != n_) {
      throw std::range_error("IwataTestFunction::Call: Input size mismatch");
    }
    auto elements = X.GetMembers();
    auto card = elements.size();
    std::size_t sum = 0;
    for (const auto& i: elements) {
      sum += i + 1;
    }
    return value_type(card * (n_ - card) + 2 * card * n_ - 5 * sum);
  }

  std::string GetName() { return "Iwata test function"; }

  std::size_t GetN() const { return n_; }
  std::size_t GetNGround() const { return n_; }

private:
  std::size_t n_;

};

}

#endif
