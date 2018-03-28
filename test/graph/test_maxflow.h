#pragma once

#include "gtest/gtest.h"

#include "core/graph/maxflow.h"
#include <vector>
#include <tuple>
#include <utility>

namespace submodular {

template <typename ValueType>
MaxflowGraph<ValueType> st_graph(std::size_t n, std::size_t s, std::size_t t,
         const std::vector<std::tuple<std::size_t, std::size_t, ValueType>>& edges)
{
  MaxflowGraph<ValueType> graph;
  auto m = edges.size();
  graph.Reserve(n, m);
  for (std::size_t i = 0; i < n; ++i) {
    graph.AddNode(i);
  }
  auto source = graph.GetNodeById(s);
  auto sink = graph.GetNodeById(t);

  for (std::size_t edge_id = 0; edge_id < m; ++edge_id) {
    std::size_t src, dst;
    ValueType cap;
    std::tie(src, dst, cap) = edges[edge_id];
    auto head = graph.GetNodeById(dst);
    auto tail = graph.GetNodeById(src);

    if (src == s && dst != t) {
      graph.AddSVArcPair(head, tail, cap, 0);
    }
    else if (src != s && dst == t) {
      graph.AddVTArcPair(head, tail, cap, 0);
    }
    else if (src != s && dst != t) {
      graph.AddArcPair(head, tail, cap, 0);
    }
    else {
      graph.AddSTOffset(cap);
    }
  }
  graph.MakeGraph(source, sink);

  return graph;
}


class MaxflowGraphTest: public testing::Test {
protected:
  using EdgeListInt = std::vector<std::tuple<std::size_t, std::size_t, int>>;
  using EdgeListDouble = std::vector<std::tuple<std::size_t, std::size_t, double>>;

  // {src, dst, capacity}
  // source = 0, sink = 6
  // maxflow value = 10 + 3
  // mincut = {0, 1, 3}
  static constexpr std::size_t n_1 = 7;
  static constexpr std::size_t s_1 = 0;
  static constexpr std::size_t t_1 = 6;
  static constexpr std::size_t data_1[9][3] = {
    {0, 1, 3},
    {0, 2, 1},
    {1, 3, 3},
    {2, 3, 5},
    {2, 4, 4},
    {4, 5, 2},
    {3, 6, 2},
    {5, 6, 3},
    {0, 6, 10} // st arc
  };

  EdgeListInt edge_list_int_1;
  EdgeListDouble edge_list_double_1;

  virtual void SetUp() {
    for (const auto& sdc: data_1) {
      edge_list_int_1.push_back(std::make_tuple(sdc[0], sdc[1], (int) sdc[2]));
      edge_list_double_1.push_back(std::make_tuple(sdc[0], sdc[1], (double) sdc[2]));
    }
  }
};

constexpr std::size_t MaxflowGraphTest::n_1;
constexpr std::size_t MaxflowGraphTest::s_1;
constexpr std::size_t MaxflowGraphTest::t_1;
constexpr std::size_t MaxflowGraphTest::data_1[9][3];

TEST_F(MaxflowGraphTest, SimpleMakeGraph) {
  auto graph_int_1 = st_graph(n_1, s_1, t_1, edge_list_int_1);

  EXPECT_EQ(graph_int_1.GetNodeNumber(), n_1);

  auto source = graph_int_1.GetSourceNode();
  auto sink = graph_int_1.GetSinkNode();
  auto node = graph_int_1.GetNode(2);
  EXPECT_EQ(node->index, 2);
  EXPECT_TRUE(graph_int_1.IsInnerNode(node));
  EXPECT_FALSE(graph_int_1.IsSourceNode(node));
  EXPECT_FALSE(graph_int_1.IsSinkNode(node));

  EXPECT_EQ(source->index, s_1);
  EXPECT_FALSE(graph_int_1.IsInnerNode(source));
  EXPECT_TRUE(graph_int_1.IsSourceNode(source));
  EXPECT_FALSE(graph_int_1.IsSinkNode(source));

  EXPECT_EQ(sink->index, t_1);
  EXPECT_FALSE(graph_int_1.IsInnerNode(sink));
  EXPECT_FALSE(graph_int_1.IsSourceNode(sink));
  EXPECT_TRUE(graph_int_1.IsSinkNode(sink));

  auto state = graph_int_1.GetState();
  EXPECT_EQ(state.first_inner_index, s_1 + 1);
  EXPECT_EQ(state.first_sink_index, t_1);
  EXPECT_EQ(state.st_offset, 10);
}

TEST_F(MaxflowGraphTest, AddCardinalityInt) {
  auto graph_int_1 = st_graph(n_1, s_1, t_1, edge_list_int_1);
  auto source = graph_int_1.GetSourceNode();
  auto sink = graph_int_1.GetSinkNode();
  auto node1 = graph_int_1.GetNode(1);
  auto node3 = graph_int_1.GetNode(3);
  EXPECT_EQ(node1->sv_cap, 3);
  EXPECT_EQ(node1->vt_cap, 0);
  EXPECT_EQ(node3->sv_cap, 0);
  EXPECT_EQ(node3->vt_cap, 2);
  EXPECT_EQ(source->sv_cap, 0);
  EXPECT_EQ(source->vt_cap, 0);
  EXPECT_EQ(sink->sv_cap, 0);
  EXPECT_EQ(sink->vt_cap, 0);

  // add constant vt_cap to every inner nodes
  graph_int_1.AddCardinalityFunction(5);

  EXPECT_EQ(node1->sv_cap, 3);
  EXPECT_EQ(node1->vt_cap, 5);
  EXPECT_EQ(node3->sv_cap, 0);
  EXPECT_EQ(node3->vt_cap, 7);
  EXPECT_EQ(source->sv_cap, 0);
  EXPECT_EQ(source->vt_cap, 0);
  EXPECT_EQ(sink->sv_cap, 0);
  EXPECT_EQ(sink->vt_cap, 0);
}

TEST_F(MaxflowGraphTest, MaxPreflowInt) {
  auto graph_int_1 = st_graph(n_1, s_1, t_1, edge_list_int_1);
  int flow_value = graph_int_1.GetMaxFlowValue();
  EXPECT_EQ(flow_value, 13);
}

TEST_F(MaxflowGraphTest, NegativeTerminalWeight) {
  auto graph_int_1 = st_graph(n_1, s_1, t_1, edge_list_int_1);

  // terminal arcs (s->v / v->t / s->t) can have negative capacities
  int multiplier = -2;
  graph_int_1.AddCardinalityFunction(multiplier);
  int offset = 10;
  int additional_offset = -30;
  int constant = offset + additional_offset;
  graph_int_1.AddSTOffset(additional_offset);


  EXPECT_EQ(graph_int_1.GetCutValueByIds({}), 4 + 0 + constant);
  EXPECT_EQ(graph_int_1.GetCutValueByIds({1}), 4 + multiplier * 1 + constant);
  EXPECT_EQ(graph_int_1.GetCutValueByIds({2}), 12 + multiplier * 1 + constant);
  EXPECT_EQ(graph_int_1.GetCutValueByIds({1, 2}), 12 + multiplier * 2 + constant);
  EXPECT_EQ(graph_int_1.GetCutValueByIds({1, 2, 3}), 6 + multiplier * 3 + constant);
  EXPECT_EQ(graph_int_1.GetCutValueByIds({1, 3}), 3 + multiplier * 2 + constant);
  EXPECT_EQ(graph_int_1.GetCutValueByIds({1, 2, 3, 4}), 4 + multiplier * 4 + constant);
  EXPECT_EQ(graph_int_1.GetCutValueByIds({1, 2, 3, 4, 5}), 5 + multiplier * 5 + constant); //mincut

  graph_int_1.FindMinCut();
  EXPECT_EQ(graph_int_1.WhatSegment(0), SOURCE);
  EXPECT_EQ(graph_int_1.WhatSegment(1), SOURCE);
  EXPECT_EQ(graph_int_1.WhatSegment(2), SOURCE);
  EXPECT_EQ(graph_int_1.WhatSegment(3), SOURCE);
  EXPECT_EQ(graph_int_1.WhatSegment(4), SOURCE);
  EXPECT_EQ(graph_int_1.WhatSegment(5), SOURCE);
  EXPECT_EQ(graph_int_1.WhatSegment(6), SINK);
}

TEST_F(MaxflowGraphTest, MaxPreflowDouble) {
  auto graph_double_1 = st_graph(n_1, s_1, t_1, edge_list_double_1);
  graph_double_1.SetTol(1e-8);
  double flow_value = graph_double_1.GetMaxFlowValue();
  EXPECT_DOUBLE_EQ(flow_value, 13);
}

TEST_F(MaxflowGraphTest, MinCutInt) {
  auto graph_int_1 = st_graph(n_1, s_1, t_1, edge_list_int_1);
  graph_int_1.FindMinCut();
  EXPECT_EQ(graph_int_1.WhatSegment(0), SOURCE);
  EXPECT_EQ(graph_int_1.WhatSegment(1), SOURCE);
  EXPECT_EQ(graph_int_1.WhatSegment(2), SINK);
  EXPECT_EQ(graph_int_1.WhatSegment(3), SOURCE);
  EXPECT_EQ(graph_int_1.WhatSegment(4), SINK);
  EXPECT_EQ(graph_int_1.WhatSegment(5), SINK);
  EXPECT_EQ(graph_int_1.WhatSegment(6), SINK);
}

TEST_F(MaxflowGraphTest, MinCutDouble) {
  auto graph_double_1 = st_graph(n_1, s_1, t_1, edge_list_double_1);
  graph_double_1.SetTol(1e-8);
  graph_double_1.FindMinCut();
  EXPECT_EQ(graph_double_1.WhatSegment(0), SOURCE);
  EXPECT_EQ(graph_double_1.WhatSegment(1), SOURCE);
  EXPECT_EQ(graph_double_1.WhatSegment(2), SINK);
  EXPECT_EQ(graph_double_1.WhatSegment(3), SOURCE);
  EXPECT_EQ(graph_double_1.WhatSegment(4), SINK);
  EXPECT_EQ(graph_double_1.WhatSegment(5), SINK);
  EXPECT_EQ(graph_double_1.WhatSegment(6), SINK);
}

TEST_F(MaxflowGraphTest, GetCutValueByIds) {
  auto graph_int_1 = st_graph(n_1, s_1, t_1, edge_list_int_1);
  int offset = 10;
  EXPECT_EQ(graph_int_1.GetCutValueByIds({}), 4 + offset);
  EXPECT_EQ(graph_int_1.GetCutValueByIds({1}), 4 + offset);
  EXPECT_EQ(graph_int_1.GetCutValueByIds({2}), 12 + offset);
  EXPECT_EQ(graph_int_1.GetCutValueByIds({1, 2}), 12 + offset);
  EXPECT_EQ(graph_int_1.GetCutValueByIds({1, 2, 3}), 6 + offset);
  EXPECT_EQ(graph_int_1.GetCutValueByIds({1, 3}), 3 + offset);
  EXPECT_EQ(graph_int_1.GetCutValueByIds({1, 2, 3, 4}), 4 + offset);
}

TEST_F(MaxflowGraphTest, Reduction) {
  auto graph = st_graph(n_1, s_1, t_1, edge_list_int_1);
  int offset = 10;
  auto state = graph.GetState();
  //graph.PrintAdjacencyList();
  EXPECT_EQ(graph.GetNodeNumber(), n_1);
  EXPECT_EQ(graph.GetCutValueByIds({1}), 4 + offset);
  EXPECT_EQ(graph.GetCutValueByIds({1, 3}), 3 + offset);

  graph.ReductionByIds({1, 3});
  //graph.PrintAdjacencyList();
  EXPECT_EQ(graph.GetNodeNumber(), 4); // {0, 1, 3, 6}
  EXPECT_TRUE(graph.IsInnerNode(graph.GetNode(1)));
  EXPECT_TRUE(graph.IsInnerNode(graph.GetNode(3)));
  EXPECT_FALSE(graph.IsInnerNode(graph.GetNode(2)));
  EXPECT_EQ(graph.GetCutValueByIds({1}), 4 + offset);
  EXPECT_EQ(graph.GetCutValueByIds({1, 3}), 3 + offset);

  graph.RestoreState(state);
  //graph.PrintAdjacencyList();
  EXPECT_EQ(graph.GetNodeNumber(), n_1);
  EXPECT_EQ(graph.GetCutValueByIds({1}), 4 + offset);
  EXPECT_EQ(graph.GetCutValueByIds({1, 3}), 3 + offset);
}

TEST_F(MaxflowGraphTest, Contraction) {
  auto graph = st_graph(n_1, s_1, t_1, edge_list_int_1);
  int offset = 10;
  auto state = graph.GetState();
  //graph.InnerTest();
  //graph.PrintAdjacencyList();
  EXPECT_EQ(graph.GetNodeNumber(), n_1);
  EXPECT_EQ(graph.GetCutValueByIds({2}), 12 + offset);
  EXPECT_EQ(graph.GetCutValueByIds({2, 4}), 10 + offset);

  int additional_offset = 20;
  graph.ContractionByIds({1, 3}, additional_offset);
  //graph.InnerTest();
  //graph.PrintAdjacencyList();
  EXPECT_EQ(graph.GetNodeNumber(), 5); // {0, 2, 4, 5, 6}
  EXPECT_TRUE(graph.IsInnerNode(graph.GetNode(2)));
  EXPECT_TRUE(graph.IsInnerNode(graph.GetNode(4)));
  EXPECT_FALSE(graph.IsInnerNode(graph.GetNode(1)));
  EXPECT_EQ(graph.GetCutValueByIds({2}), 6 + offset + additional_offset);
  EXPECT_EQ(graph.GetCutValueByIds({2, 4}), 4 + offset + additional_offset);

  graph.RestoreState(state);
  //graph.PrintAdjacencyList();
  EXPECT_EQ(graph.GetNodeNumber(), n_1);
  EXPECT_EQ(graph.GetCutValueByIds({2}), 12 + offset);
  EXPECT_EQ(graph.GetCutValueByIds({2, 4}), 10 + offset);
}

TEST_F(MaxflowGraphTest, ContractionNoInnerNodes) {
  auto graph = st_graph(n_1, s_1, t_1, edge_list_int_1);
  int offset = 10;

  auto fv = graph.GetCutValueByIds({1, 2, 3, 4, 5});
  EXPECT_EQ(fv, 5 + offset);

  graph.ContractionByIds({1, 2, 3, 4, 5}, -fv);
  EXPECT_EQ(graph.GetNodeNumber(), 2); // {0, 2, 4, 5, 6}
  EXPECT_FALSE(graph.IsInnerNode(graph.GetNode(2)));
  EXPECT_FALSE(graph.IsInnerNode(graph.GetNode(4)));
  EXPECT_FALSE(graph.IsInnerNode(graph.GetNode(1)));

  auto flow = graph.GetMaxFlowValue();
  EXPECT_EQ(flow, 0);
}

}
