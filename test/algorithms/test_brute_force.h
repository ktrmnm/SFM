#pragma once

#include <string>
#include <iostream>
#include "gtest/gtest.h"
#include "core/set_utils.h"
#include "core/oracles/modular.h"
#include "core/oracles/iwata_test_function.h"
#include "core/algorithms/brute_force.h"

namespace submodular {

TEST(BruteForce, Modular) {
  ModularOracle<int> F1({ -1, -2, -3, 4, 5 });
  BruteForce<int> solver1;
  solver1.Minimize(F1);
  auto X1 = solver1.GetMinimizer();
  auto val1 = solver1.GetMinimumValue();
  EXPECT_EQ(X1, Set(std::string("11100")));
  EXPECT_EQ(val1, -6);
  std::cout << solver1.GetReporter() << std::endl;

  ModularOracle<int> F2({ 1, 1, 1, -1, -1, 1, 1, -1, 1});
  BruteForce<int> solver2;
  solver2.Minimize(F2);
  auto X2 = solver2.GetMinimizer();
  auto val2 = solver2.GetMinimumValue();
  EXPECT_EQ(X2, Set(std::string("000110010")));
  EXPECT_EQ(val2, -3);
  std::cout << solver2.GetReporter() << std::endl;
}

TEST(BruteForce, IwataTestFunction) {
  IwataTestFunction<int> F1(5); // F(2345) = -26 is the minimum value
  BruteForce<int> solver1;
  solver1.Minimize(F1);
  EXPECT_EQ(solver1.GetMinimumValue(), -26);
  EXPECT_EQ(solver1.GetMinimizer(), Set(std::string("01111")));
  std::cout << solver1.GetReporter() << std::endl;
}

}
