#pragma once

#include "gtest/gtest.h"

#include "core/graph/stcut.h"
#include "core/graph/divide_conquer.h"
#include <vector>
#include <utility>

namespace submodular {

class DivideConquerTest: public testing::Test {
protected:

  static constexpr std::size_t n = 5;
  static constexpr std::size_t edges[4][2] = {
    {0, 1},
    {1, 2},
    {2, 3},
    {3, 4}
  };
  static constexpr double caps[4] = { 1, 1, 1, 1 };
  static constexpr double modular_dec[5] = { 4, 3, 2, 1, 0 };
  static constexpr double modular_inc[5] = { 0, 1, 2, 3, 4 };

  using EdgeList = std::vector<std::pair<std::size_t, std::size_t>>;
  EdgeList edge_list;
  std::vector<double> capacities;
  std::vector<double> x_dec_1;
  std::vector<double> x_dec_2;
  std::vector<double> x_inc_1;

  virtual void SetUp() {
    for (const auto& sd: edges) {
      edge_list.push_back(std::make_pair(sd[0], sd[1]));
    }

    for (const auto& c: caps) {
      capacities.push_back(c);
    }

    for (int i = 0; i < n; ++i) {
      x_dec_1.push_back(- modular_dec[i]);
      x_dec_2.push_back(0.25 * (- modular_dec[i]));
      x_inc_1.push_back(- modular_inc[i]);
    }

  }
};

constexpr std::size_t DivideConquerTest::n;
constexpr std::size_t DivideConquerTest::edges[4][2];
constexpr double DivideConquerTest::caps[4];
constexpr double DivideConquerTest::modular_dec[5];
constexpr double DivideConquerTest::modular_inc[5];

TEST_F(DivideConquerTest, DecreasingCase1) {
  auto F = CutPlusModular<double>::FromEdgeList(
    n, DIRECTED, edge_list, capacities, x_dec_1
  );
  DivideConquerMNP<double> solver;
  auto mnp = solver.Solve(F);

  EXPECT_DOUBLE_EQ(-mnp[0], 3);
  EXPECT_DOUBLE_EQ(-mnp[1], 3);
  EXPECT_DOUBLE_EQ(-mnp[2], 2);
  EXPECT_DOUBLE_EQ(-mnp[3], 1);
  EXPECT_DOUBLE_EQ(-mnp[4], 1);
}

TEST_F(DivideConquerTest, DecreasingCase2) {

  auto F = CutPlusModular<double>::FromEdgeList(
    n, DIRECTED, edge_list, capacities, x_dec_2
  );
  DivideConquerMNP<double> solver;
  auto mnp = solver.Solve(F);

  EXPECT_DOUBLE_EQ(-mnp[0], 0.5);
  EXPECT_DOUBLE_EQ(-mnp[1], 0.5);
  EXPECT_DOUBLE_EQ(-mnp[2], 0.5);
  EXPECT_DOUBLE_EQ(-mnp[3], 0.5);
  EXPECT_DOUBLE_EQ(-mnp[4], 0.5);
}

TEST_F(DivideConquerTest, IncreasingCase) {
  auto F = CutPlusModular<double>::FromEdgeList(
    n, DIRECTED, edge_list, capacities, x_inc_1
  );
  DivideConquerMNP<double> solver;
  auto mnp = solver.Solve(F);

  EXPECT_DOUBLE_EQ(-mnp[0], 0);
  EXPECT_DOUBLE_EQ(-mnp[1], 1);
  EXPECT_DOUBLE_EQ(-mnp[2], 2);
  EXPECT_DOUBLE_EQ(-mnp[3], 3);
  EXPECT_DOUBLE_EQ(-mnp[4], 4);
}

}
