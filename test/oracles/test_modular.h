#pragma once

#include "gtest/gtest.h"

#include <vector>
#include <string>
#include <stdexcept>
#include "core/set_utils.h"
#include "core/oracles/modular.h"

namespace submodular {

TEST(ModularOracle, Call) {
  //std::vector<int> x = { 1, 2, 3, 4, 5 };
  ModularOracle<int> F({ 1, 2, 3, 4, 5 });
  EXPECT_EQ(F.Call(Set(std::string("00000"))), 0);
  EXPECT_EQ(F.Call(Set(std::string("10000"))), 1);
  EXPECT_EQ(F.Call(Set(std::string("11000"))), 3);
  EXPECT_EQ(F.Call(Set(std::string("11100"))), 6);
  EXPECT_EQ(F.Call(Set(std::string("11110"))), 10);
  EXPECT_EQ(F.Call(Set(std::string("11111"))), 15);
}

TEST(ModularOracle, RangeError) {
  std::vector<int> x = { 1, 2, 3, 4, 5 };
  ModularOracle<int> F(x);
  EXPECT_THROW(F.Call(Set(std::string("101010"))), std::range_error);
}

TEST(ConstantOracle, Call) {
  ConstantOracle<int> F(5, 1);
  EXPECT_EQ(F.Call(Set(std::string("00000"))), 1);
  EXPECT_EQ(F.Call(Set(std::string("10000"))), 1);
  EXPECT_EQ(F.Call(Set(std::string("11000"))), 1);
  EXPECT_EQ(F.Call(Set(std::string("11100"))), 1);
  EXPECT_EQ(F.Call(Set(std::string("11110"))), 1);
  EXPECT_EQ(F.Call(Set(std::string("11111"))), 1);
}


}
