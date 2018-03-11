#pragma once

#include "gtest/gtest.h"

#include <vector>
#include "core/set_utils.h"
#include "core/base.h"
#include "core/oracles/modular.h"
#include "core/oracles/iwata_test_function.h"

namespace submodular {

TEST(BaseMisc, GetDescendingOrder) {
  std::vector<int> x1 = { 3, 4, 2, 0, 1 };
  auto ord1 = GetDescendingOrder(x1);
  EXPECT_EQ(ord1[0], 1);
  EXPECT_EQ(ord1[1], 0);
  EXPECT_EQ(ord1[2], 2);
  EXPECT_EQ(ord1[3], 4);
  EXPECT_EQ(ord1[4], 3);

  std::vector<double> x2 = { 2, 0, 3, 1, 4 };
  auto ord2 = GetDescendingOrder(x2);
  EXPECT_EQ(ord2[0], 4);
  EXPECT_EQ(ord2[1], 2);
  EXPECT_EQ(ord2[2], 0);
  EXPECT_EQ(ord2[3], 3);
  EXPECT_EQ(ord2[4], 1);
}

TEST(GreedyBase, Modular) {
  ModularOracle<int> F({1, 2, 3, 4, 5});
  // Since base polytope of a modular function is a single point,
  // GreedyBase does not depend on the order.
  OrderType ord1 = {0, 1, 2, 3, 4};
  OrderType ord2 = {4, 2, 3, 1, 0};
  auto base1 = GreedyBase(F, ord1);
  auto base2 = GreedyBase(F, ord2);

  EXPECT_EQ(base1[0], 1);
  EXPECT_EQ(base1[1], 2);
  EXPECT_EQ(base1[2], 3);
  EXPECT_EQ(base1[3], 4);
  EXPECT_EQ(base1[4], 5);

  EXPECT_EQ(base2[0], 1);
  EXPECT_EQ(base2[1], 2);
  EXPECT_EQ(base2[2], 3);
  EXPECT_EQ(base2[3], 4);
  EXPECT_EQ(base2[4], 5);
}

TEST(GreedyBase, IwataTestFunction) {
  IwataTestFunction<int> F(5);

  OrderType ord1 = {0, 1, 2, 3, 4}; // F(0) = 9, F(01) = 11, F(012) = 6, F(0123) = -6, F(01234) = -25
  auto base1 = GreedyBase(F, ord1);

  EXPECT_EQ(base1[0], 9);
  EXPECT_EQ(base1[1], 11 - 9);
  EXPECT_EQ(base1[2], 6 - 11);
  EXPECT_EQ(base1[3], - 6 - 6);
  EXPECT_EQ(base1[4], - 25 - (- 6));
}


}
