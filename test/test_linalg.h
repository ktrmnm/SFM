#pragma once

#include "gtest/gtest.h"
#include <Eigen/Core>
#include <vector>
#include <iostream>
#include "core/linalg.h"

namespace linalg {

TEST(reduce_bases, Case1){

  // matrix of rank 3
  std::vector<std::vector<double>> A = {
    {1, 4, 5, 3, 6},
    {4, 1, 1, 6, 4},
    {0, 9, 7, 7, 7},
    {5, 5, 6, 9, 10}, // A[0] + A[1]
    {4, 10, 8, 13, 11}, // A[1] + A[2]
    {1, 13, 12, 10, 13}, // A[0] + A[2]
    {-3, 3, 4, -3, 2}, // A[0] - A[1]
    {4, -8, -6, -1, -3}, // A[1] - A[2]
    {-1, 5, 2, 4, 1}, // A[2] - A[0]
    {0, 0, 0, 0, 0}
  };

  // x is obtained by 0.25 * A[0] + 0.25 * A[1] + 0.5 * A[2]
  std::vector<double> x = { 1.25, 5.75, 5.0, 5.75, 6.0 };

  // A * coef = x
  std::vector<double> coef = { 0.05, 0.05, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0 };
  auto A_mxd = stdvecvec2mxd(5, A.size(), A);
  auto coef_vxd = stdvec2vxd(coef.size(), coef);
  auto Ac = A_mxd * coef_vxd;
  EXPECT_DOUBLE_EQ(Ac(0), x[0]);
  EXPECT_DOUBLE_EQ(Ac(1), x[1]);
  EXPECT_DOUBLE_EQ(Ac(2), x[2]);
  EXPECT_DOUBLE_EQ(Ac(3), x[3]);
  EXPECT_DOUBLE_EQ(Ac(4), x[4]);

  auto m = reduce_bases(5, A, coef);
  EXPECT_LE(m, 10);

  double sum = 0;
  for (const auto& a: coef) {
    sum += a;
  }
  EXPECT_DOUBLE_EQ(sum, 1);

  auto A_mxd_reduced = stdvecvec2mxd(5, A.size(), A);
  auto coef_vxd_reduced = stdvec2vxd(coef.size(), coef);
  auto Ac_reduced = A_mxd_reduced * coef_vxd_reduced;

  EXPECT_DOUBLE_EQ(Ac_reduced(0), x[0]);
  EXPECT_DOUBLE_EQ(Ac_reduced(1), x[1]);
  EXPECT_DOUBLE_EQ(Ac_reduced(2), x[2]);
  EXPECT_DOUBLE_EQ(Ac_reduced(3), x[3]);
  EXPECT_DOUBLE_EQ(Ac_reduced(4), x[4]);

}

TEST(reduce_bases, Case2){

  // randomly generated matrix
  std::vector<std::vector<double>> A = {
    { -4,  -5,   8,  -7,  -2,   0},
    {  9,   1,  -8,   9,  -8,  -2},
    {  0,  -6,   2,   0,  -5,   7},
    {  8,   6,  -5,  -9,   4,  -8},
    {  3,   1,   5,   7,   9,   9},
    {  4,   6,  -9,   5,   5,  -9},
    { -6,  -4,   5,   0,   3,  -9},
    { -9,   1,   6,  -9,  -7,   1},
    {  2,   6,   8,  -7,   9,   8},
    {  5,   4,  -7,   1,   7,   8},
    {  6,  -4,  -5,   6,   2,   6},
    { -2,  -9,  -4,  -5,   7,  -6},
    { -1,   5,   0,   4,  -6,  -9},
    { -8,   7,   4,  -4,  -5,  -5},
    {  5,  -3,   1,  -2,   4,   5},
    { -3,   9,  -1,  -7, -10,   6},
    { -7,   8,  -6,  -9,   4,   7},
    { -6,   7,   7,  -2,  -3,   3},
    {  8,  -8,   8,   8,  -7,   4},
    {  6,   8, -10,   6,   1,   8}
  };

  std::vector<double> coef = {
    4,   3, -10,  -1,  -8,  -8,   7, -10,  -9, -10,
    0,   5,  -5,  -6,   6,   6,   1,  -4,  -3,  -7
  };
  std::vector<double> x = { -67, -279,  -34,  -87,  -69, -255}; // A * coef

  auto m_before = A.size();
  auto m_after = reduce_bases(6, A, coef);
  EXPECT_LE(m_after, m_before);

  auto A_mxd_reduced = stdvecvec2mxd(6, A.size(), A);
  auto coef_vxd_reduced = stdvec2vxd(coef.size(), coef);
  auto Ac_reduced = A_mxd_reduced * coef_vxd_reduced;

  double tol = 1e-8;
  EXPECT_NEAR(Ac_reduced(0), x[0], tol);
  EXPECT_NEAR(Ac_reduced(1), x[1], tol);
  EXPECT_NEAR(Ac_reduced(2), x[2], tol);
  EXPECT_NEAR(Ac_reduced(3), x[3], tol);
  EXPECT_NEAR(Ac_reduced(4), x[4], tol);
  EXPECT_NEAR(Ac_reduced(5), x[5], tol);
}


}
