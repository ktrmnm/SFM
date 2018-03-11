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

TEST(Set, ConstructorB) {
  Set V(5, {2, 4});
  EXPECT_EQ(V.n_, 5);
  EXPECT_EQ(V[0], 0);
  EXPECT_EQ(V[1], 0);
  EXPECT_EQ(V[2], 1);
  EXPECT_EQ(V[3], 0);
  EXPECT_EQ(V[4], 1);

  EXPECT_THROW(new Set(5, {10}), std::range_error);
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
  EXPECT_TRUE(V1 != V2);
  V1.C();
  EXPECT_TRUE(V1 == V2);
  V2.C();
  EXPECT_TRUE(V1 != V2);

  auto V3 = ~V1;
  EXPECT_TRUE(V3 == V2);
}


}
