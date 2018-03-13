#pragma once

#include "gtest/gtest.h"

#include <stdexcept>
#include <string>
#include "core/set_utils.h"

namespace submodular {

TEST(Set, ConstructorA) {
  Set V(5);
  EXPECT_EQ(V.n_, 5);
  EXPECT_EQ(V[0], 0);
  EXPECT_EQ(V[1], 0);
  EXPECT_EQ(V[2], 0);
  EXPECT_EQ(V[3], 0);
  EXPECT_EQ(V[4], 0);
}

/*
TEST(Set, ConstructorB) {
  Set V(5, {2, 4});
  EXPECT_EQ(V.n_, 5);
  EXPECT_EQ(V[0], 0);
  EXPECT_EQ(V[1], 0);
  EXPECT_EQ(V[2], 1);
  EXPECT_EQ(V[3], 0);
  EXPECT_EQ(V[4], 1);

  //EXPECT_THROW(new Set(5, {10}), std::range_error);
  //NOTE: Set(5, {10}) matches constructor (d), and this test may fail.
  //In order to avoid such ambiguity, one of these constructors should be depricated.
  EXPECT_THROW(new Set(5, std::vector<size_t>{10}), std::range_error);
}
*/

TEST(Set, FromIndices) {
  auto V = Set::FromIndices(5, {2, 4});
  EXPECT_EQ(V.n_, 5);
  EXPECT_EQ(V[0], 0);
  EXPECT_EQ(V[1], 0);
  EXPECT_EQ(V[2], 1);
  EXPECT_EQ(V[3], 0);
  EXPECT_EQ(V[4], 1);

  EXPECT_THROW(Set::FromIndices(5, {10}), std::range_error);
}

TEST(Set, ConstructorC) {
  Set V(std::string("01010"));
  EXPECT_EQ(V.n_, 5);
  EXPECT_EQ(V[0], 0);
  EXPECT_EQ(V[1], 1);
  EXPECT_EQ(V[2], 0);
  EXPECT_EQ(V[3], 1);
  EXPECT_EQ(V[4], 0);

  EXPECT_THROW(new Set(std::string("01020")), std::invalid_argument);
}

TEST(Set, Complement) {
  Set V1(std::string("0101100"));
  Set V2(std::string("1010011"));
  /*
  EXPECT_TRUE(V1 != V2);
  V1.C();
  EXPECT_TRUE(V1 == V2);
  V2.C();
  EXPECT_TRUE(V1 != V2);

  auto V3 = ~V1;
  EXPECT_TRUE(V3 == V2);
  */
  EXPECT_TRUE(V1.Complement() == V2);
  EXPECT_TRUE(V1.Complement().Complement() == V1);
}

TEST(Partition, RemoveCell) {
  Partition p = Partition::MakeFine(5); // p = {{0}, {1}, {2}, {3}, {4}}
  EXPECT_EQ(Set(std::string("11111")), p.Expand());
  p.RemoveCell(3); // p = {{0}, {1}, {2}, {4}}
  EXPECT_EQ(Set(std::string("11101")), p.Expand());
  p.RemoveCell(0);
  EXPECT_EQ(Set(std::string("01101")), p.Expand());
  p.RemoveCell(0); // cell {0} is already removed
  EXPECT_EQ(Set(std::string("01101")), p.Expand());
}

TEST(Partition, Merge) {
  Partition p = Partition::MakeFine(5);
  p.MergeCells(0, 1);
  EXPECT_EQ(Set(std::string("11000")), p.GetCellAsSet(0));
  p.MergeCells({3, 2, 4});
  EXPECT_EQ(Set(std::string("00111")), p.GetCellAsSet(2));
  p.RemoveCell(2);
  EXPECT_EQ(Set(std::string("11000")), p.Expand());
}


}
