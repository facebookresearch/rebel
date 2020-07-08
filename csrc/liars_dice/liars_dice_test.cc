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
#include <math.h>

#include <gtest/gtest.h>

#include "liars_dice.h"

using namespace liars_dice;

class GameTest : public ::testing::Test {
 protected:
  const int num_dice = 2;
  const int num_faces = 6;
  const Game game;
  const PartialPublicState root;

  GameTest() : game(num_dice, num_faces), root(game.get_initial_state()) {}
};

TEST_F(GameTest, TestUnpacl) {
  {
    auto unpacked = game.unpack_action(0);
    ASSERT_EQ(unpacked.quantity, 1);
    ASSERT_EQ(unpacked.face, 0);
  }
  {
    auto unpacked = game.unpack_action(1);
    ASSERT_EQ(unpacked.quantity, 1);
    ASSERT_EQ(unpacked.face, 1);
  }
  {
    auto unpacked = game.unpack_action(6);
    ASSERT_EQ(unpacked.quantity, 2);
    ASSERT_EQ(unpacked.face, 0);
  }
}

TEST_F(GameTest, TestRoot) {
  ASSERT_EQ(root.player_id, 0);
  {
    auto range = game.get_bid_range(root);
    ASSERT_EQ(range.first, 0);
    ASSERT_EQ(range.second, 4 * 6);
  }
  {
    auto range = game.get_bid_range(game.act(root, 0));
    ASSERT_EQ(range.first, 1);
    ASSERT_EQ(range.second, 4 * 6 + 1);
  }
  {
    auto state = game.act(root, 0);
    auto range = game.get_bid_range(state);
    ASSERT_EQ(range.first, 1);
    ASSERT_EQ(range.second, 4 * 6 + 1);
  }
  {
    auto state = game.act(root, 11);
    auto range = game.get_bid_range(state);
    ASSERT_EQ(range.first, 11 + 1);
    ASSERT_EQ(range.second, 4 * 6 + 1);
  }
  {
    auto state = game.act(game.act(root, 0), game.liar_call());
    auto range = game.get_bid_range(state);
    ASSERT_EQ(range.first, 4 * 6 + 1);
    ASSERT_EQ(range.second, 4 * 6 + 1);
  }
}

TEST_F(GameTest, TestPlayerSequencw) {
  auto state = root;
  for (int i = 0; i < 4 * 6 + 1; ++i) {
    state = game.act(state, i);
    ASSERT_EQ(state.player_id, (i + 1) % 2);
  }
}

TEST_F(GameTest, TestNumMatchesSimple) {
  // Hand: 2 1's.
  auto num_matches = game.num_matches(0);
  ASSERT_EQ(num_matches, (std::vector<int>{2, 0, 0, 0, 0, 0}));
}

TEST_F(GameTest, TestNumMatchesWild) {
  // Hand: 2 6's.
  ASSERT_EQ(game.wild_face(), 5);
  auto num_matches = game.num_matches(game.num_hands() - 1);
  ASSERT_EQ(num_matches, (std::vector<int>{2, 2, 2, 2, 2, 2}));
}

TEST_F(GameTest, TestNumMatchesSemiWild) {
  // Hand: 1 and 6's.
  auto num_matches = game.num_matches(0 * 6 + 5);
  ASSERT_EQ(num_matches, (std::vector<int>{2, 1, 1, 1, 1, 1}));
}