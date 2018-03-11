#pragma once

#include "gtest/gtest.h"

#include <vector>
#include <string>
#include "core/set_utils.h"
#include "core/oracles/iwata_test_function.h"

namespace submodular {

TEST(IwataTestFunction, Call) {
  IwataTestFunction<int> F(5);
  EXPECT_EQ(F.Call(Set(std::string("00000"))), 0);
  EXPECT_EQ(F.Call(Set(std::string("10000"))), 9);
  EXPECT_EQ(F.Call(Set(std::string("11000"))), 11);
  EXPECT_EQ(F.Call(Set(std::string("11100"))), 6);
  EXPECT_EQ(F.Call(Set(std::string("11110"))), -6);
  EXPECT_EQ(F.Call(Set(std::string("11111"))), -25);
}

}
