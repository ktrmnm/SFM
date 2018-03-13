#pragma once

#include "gtest/gtest.h"

#include <vector>
#include <string>
//#include <stdexcept>
#include <memory>
#include "core/set_utils.h"
#include "core/oracle.h"
#include "core/oracles/modular.h"

namespace submodular {

TEST(ReducibleOracle, ModularReduction) {
  ModularOracle<int> modular({ 1, 2, 3, 4, 5 });
  ReducibleOracle<int> F1(modular);
  Set X(std::string("11000"));
  EXPECT_EQ(modular.Call(X), F1.Call(X));
  EXPECT_EQ(F1.GetNGround(), 5);

  auto F2 = F1.ReductionCopy(X);
  EXPECT_EQ(modular.Call(X), F2.Call(Set(std::string("11"))));
  EXPECT_EQ(F2.GetNGround(), 5);
  EXPECT_EQ(F2.GetN(), 2);
}

TEST(ReducibleOracle, ModularContraction) {
  ModularOracle<int> modular({ 1, 2, 3, 4, 5 });
  ReducibleOracle<int> F1(modular);
  Set X1(std::string("11000"));
  EXPECT_EQ(modular.Call(X1), F1.Call(X1));
  EXPECT_EQ(F1.GetNGround(), 5);
  EXPECT_EQ(F1.GetN(), 5);

  auto F2 = F1.ContractionCopy(X1);
  Set X2(std::string("00110"));
  EXPECT_EQ(modular.Call(X1.Union(X2)) - modular.Call(X1), F2.Call(X2));
  EXPECT_EQ(F2.GetNGround(), 5);
  EXPECT_EQ(F2.GetN(), 3);
}

TEST(ReducibleOracle, ModularShrink) {
  ModularOracle<int> modular({ 1, 2, 3, 4, 5 });
  ReducibleOracle<int> F1(modular);
  Set X1(std::string("11000"));
  EXPECT_EQ(modular.Call(X1), F1.Call(X1));
  EXPECT_EQ(F1.GetNGround(), 5);

  auto F2 = F1.ShrinkNodesCopy(X1);
  Set X2(std::string("10000"));
  EXPECT_EQ(modular.Call(X1), F2.Call(X2));
  EXPECT_EQ(F2.GetNGround(), 5);
  EXPECT_EQ(F2.GetN(), 4);
}

}
