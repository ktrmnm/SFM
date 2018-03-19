#pragma once

#include <string>
#include <iostream>
#include "gtest/gtest.h"
#include "core/set_utils.h"
#include "core/oracles/modular.h"
#include "core/oracles/iwata_test_function.h"
#include "core/algorithms/sfm_iff.h"

namespace submodular {

TEST(IFFWP, Modular) {
  ReducibleOracle<int> F1(ModularOracle<int>({ -1, -2, -3, 4, 5 }));
  IFFWP<int> solver1;
  solver1.Minimize(F1);
  auto X1 = solver1.GetMinimizer();
  auto val1 = solver1.GetMinimumValue();
  EXPECT_EQ(X1, Set(std::string("11100")));
  EXPECT_EQ(val1, -6);

  ReducibleOracle<int> F2(ModularOracle<int>({ 1, 1, 1, -1, -1, 1, 1, -1, 1}));
  IFFWP<int> solver2;
  solver2.Minimize(F2);
  auto X2 = solver2.GetMinimizer();
  auto val2 = solver2.GetMinimumValue();
  EXPECT_EQ(X2, Set(std::string("000110010")));
  EXPECT_EQ(val2, -3);
}

TEST(IFFWP, IwataTestFunction) {
  ReducibleOracle<int> F1(IwataTestFunction<int>(5)); // F(2345) = -26 is the minimum value
  IFFWP<int> solver1;
  solver1.Minimize(F1);
  EXPECT_EQ(solver1.GetMinimumValue(), -26);
  EXPECT_EQ(solver1.GetMinimizer(), Set(std::string("01111")));
}

}
