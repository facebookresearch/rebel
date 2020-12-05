// Copyright (c) Facebook, Inc. and its affiliates.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tree.h"
#include <gtest/gtest.h>

using namespace liars_dice;

TEST(TreeTest, TestUnroll) {
  const int num_dice = 1;
  const int num_faces = 2;
  Game game(num_dice, num_faces);

  auto nodes = unroll_tree(game);

  ASSERT_EQ(nodes.size(), 31);
  EXPECT_EQ(nodes[0].get_children(), (std::vector<int>{1, 2, 3, 4}));
  EXPECT_EQ(nodes[1].get_children(), (std::vector<int>{5, 6, 7, 8}));
  EXPECT_EQ(nodes[2].get_children(), (std::vector<int>{9, 10, 11}));
  EXPECT_EQ(nodes[15].get_children(), (std::vector<int>{25, 26}));
  EXPECT_EQ(nodes[16].get_children(), (std::vector<int>{27}));
  EXPECT_EQ(nodes[25].get_children(), (std::vector<int>{30}));
}

TEST(TreeTest, TestUnrollDepthZero) {
  const int num_dice = 2;
  const int num_faces = 6;
  Game game(num_dice, num_faces);

  const int last_bid = 22;
  ASSERT_EQ(game.action_to_string(last_bid), "bid(quantity=4,face=4)");
  const auto root = PartialPublicState{last_bid, 0};

  auto nodes = unroll_tree(game, root, 0);

  ASSERT_EQ(nodes.size(), 1);
  EXPECT_EQ(nodes[0].parent, -1);
  EXPECT_EQ(nodes[0].get_children().size(), 0);
  EXPECT_EQ(nodes[0].state, root);
}

TEST(TreeTest, TestUnrollDepthOne) {
  const int num_dice = 2;
  const int num_faces = 6;
  Game game(num_dice, num_faces);

  const int last_bid = 22;
  ASSERT_EQ(game.action_to_string(last_bid), "bid(quantity=4,face=4)");
  const auto root = PartialPublicState{last_bid, 0};

  auto nodes = unroll_tree(game, root, 1);

  ASSERT_EQ(nodes.size(), 3);
  EXPECT_EQ(nodes[0].parent, -1);
  EXPECT_EQ(nodes[0].get_children(), (std::vector<int>{1, 2}));
  EXPECT_EQ(nodes[1].parent, 0);
  EXPECT_EQ(nodes[2].parent, 0);
}

TEST(TreeTest, TestUnrollDepthTwo) {
  const int num_dice = 2;
  const int num_faces = 6;
  Game game(num_dice, num_faces);

  const int last_bid = 22;
  ASSERT_EQ(game.action_to_string(last_bid), "bid(quantity=4,face=4)");
  const auto root = PartialPublicState{last_bid, 0};

  auto nodes = unroll_tree(game, root, 2);

  ASSERT_EQ(nodes.size(), 4);
  EXPECT_EQ(nodes[0].parent, -1);
  EXPECT_EQ(nodes[0].get_children(), (std::vector<int>{1, 2}));
  EXPECT_EQ(nodes[1].parent, 0);
  EXPECT_EQ(nodes[2].parent, 0);
  EXPECT_EQ(nodes[3].parent, 1);
}

TEST(TreeTest, TestUnrollDepthTwoDeep) {
  const int num_dice = 2;
  const int num_faces = 6;
  Game game(num_dice, num_faces);

  const int last_bid = 21;
  ASSERT_EQ(game.action_to_string(last_bid), "bid(quantity=4,face=3)");
  const auto root = PartialPublicState{last_bid, 0};

  auto nodes = unroll_tree(game, root, 2);

  ASSERT_EQ(nodes.size(), 7);
  EXPECT_EQ(nodes[0].get_children(), (std::vector<int>{1, 2, 3}));
  EXPECT_EQ(nodes[1].get_children(), (std::vector<int>{4, 5}));
  EXPECT_EQ(nodes[2].get_children(), (std::vector<int>{6}));
}

TEST(TreeTest, TestTreeIsBreadthFirst) {
  // Required for partial initialization to work.
  const int num_dice = 1;
  const int num_faces = 5;
  const Game game(num_dice, num_faces);
  const auto tree = unroll_tree(game);
  for (size_t subtree_depth = 0; subtree_depth < 20; ++subtree_depth) {
    const auto subtree =
        unroll_tree(game, game.get_initial_state(), subtree_depth);
    for (size_t i = 0; i < subtree.size(); ++i) {
      ASSERT_EQ(tree[i].state, subtree[i].state);
      if (subtree[i].num_children()) {
        ASSERT_EQ(tree[i].children_begin, subtree[i].children_begin);
        ASSERT_EQ(tree[i].children_end, subtree[i].children_end);
        ASSERT_EQ(tree[i].parent, subtree[i].parent);
      }
    }
  }
}
