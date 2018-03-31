#pragma once

#include <string>
#include <iostream>
#include "gtest/gtest.h"
#include "core/set_utils.h"
#include "core/oracles/modular.h"
#include "core/oracles/iwata_test_function.h"
#include "core/algorithms/sfm_fw.h"

namespace submodular {

TEST(FWRobust, Modular) {
  ModularOracle<int> F1({ -1, -2, -3, 4, 5 });
  FWRobust<int> solver1;
  solver1.Minimize(F1);
  auto X1 = solver1.GetMinimizer();
  auto val1 = solver1.GetMinimumValue();
  EXPECT_EQ(X1, Set(std::string("11100")));
  EXPECT_EQ(val1, -6);
  auto r1 = solver1.GetReporter();
  std::cout << r1 << std::endl;

  ModularOracle<int> F2({ 1, 1, 1, -1, -1, 1, 1, -1, 1});
  FWRobust<int> solver2;
  solver2.Minimize(F2);
  auto X2 = solver2.GetMinimizer();
  auto val2 = solver2.GetMinimumValue();
  EXPECT_EQ(X2, Set(std::string("000110010")));
  EXPECT_EQ(val2, -3);
  auto r2 = solver2.GetReporter();
  std::cout << r2 << std::endl;
}

TEST(FWRobust, IwataTestFunction5) {
  IwataTestFunction<int> F1(5); // F(2345) = -26 is the minimum value
  FWRobust<int> solver1;
  solver1.Minimize(F1);
  EXPECT_EQ(solver1.GetMinimumValue(), -26);
  EXPECT_EQ(solver1.GetMinimizer(), Set(std::string("01111")));
  auto r1 = solver1.GetReporter();
  std::cout << r1 << std::endl;
}

TEST(FWRobust, IwataTestFunction10) {
  IwataTestFunction<int> F1(10);
  FWRobust<int> solver1;
  BruteForce<int> solver2;
  solver1.Minimize(F1);
  solver2.Minimize(F1);
  std::cout << solver1.GetReporter() << std::endl;
  std::cout << solver2.GetReporter() << std::endl;
  EXPECT_EQ(solver1.GetMinimumValue(), solver2.GetMinimumValue());
}

TEST(FWRobust, IwataTestFunctionDouble10) {
  IwataTestFunction<double> F1(10);
  FWRobust<double> solver1;
  BruteForce<double> solver2;
  solver1.Minimize(F1);
  solver2.Minimize(F1);
  std::cout << solver1.GetReporter() << std::endl;
  std::cout << solver2.GetReporter() << std::endl;
  EXPECT_EQ(solver1.GetMinimumValue(), solver2.GetMinimumValue());
}

/*
TEST(FWRobust, IwataTestFunction20) {
  IwataTestFunction<int> F1(20);
  FWRobust<int> solver1;
  BruteForce<int> solver2;
  solver1.Minimize(F1);
  solver2.Minimize(F1);
  std::cout << solver1.GetReporter() << std::endl;
  std::cout << solver2.GetReporter() << std::endl;
  EXPECT_EQ(solver1.GetMinimumValue(), solver2.GetMinimumValue());
}
*/

TEST(FWRobust, IwataTestFunction30) {
  IwataTestFunction<int> F1(30);
  FWRobust<int> solver1;
  //BruteForce<int> solver2;
  solver1.Minimize(F1);
  //solver2.Minimize(F1);
  std::cout << solver1.GetReporter() << std::endl;
  //std::cout << solver2.GetReporter() << std::endl;
  EXPECT_EQ(solver1.GetMinimumValue(), -651);
}

}
