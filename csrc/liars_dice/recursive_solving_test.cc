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

#include "real_net.h"
#include "recursive_solving.h"

using namespace liars_dice;

TEST(Mdp, TestZeroNet) {
  const int num_dice = 1;
  const int num_faces = 3;
  SubgameSolvingParams params;
  params.num_iters = 100;
  params.max_depth = 1;
  params.linear_update = true;
  const Game game(num_dice, num_faces);
  auto net = create_zero_net(game.num_hands());
  RlRunner runner(game, params, net, /*seed=*/0);

  for (int i = 0; i < 10; ++i) {
    runner.step();
  }
}

TEST(Mdp, TestZeroNetSampleLeaf) {
  RecursiveSolvingParams params;
  params.subgame_params.num_iters = 100;
  params.subgame_params.max_depth = 2;
  params.subgame_params.linear_update = true;
  params.sample_leaf = true;
  params.num_dice = 1;
  params.num_faces = 3;
  const Game game(params.num_dice, params.num_faces);
  auto net = create_zero_net(game.num_hands());
  RlRunner runner(params, net, /*seed=*/0);

  for (int i = 0; i < 10; ++i) {
    runner.step();
  }
}

TEST(Mdp, TestZeroNetComputeStrategy) {
  const int num_dice = 1;
  const int num_faces = 3;
  SubgameSolvingParams params;
  params.num_iters = 100;
  params.max_depth = 1;
  params.linear_update = true;
  const Game game(num_dice, num_faces);
  auto net = create_zero_net(game.num_hands());

  auto strategy = compute_strategy_recursive(game, params, net);
  auto full_tree = unroll_tree(game);
  ASSERT_EQ(strategy.size(), full_tree.size());
}

TEST(Mdp, TestZeroNetComputeStrategyToLeaf) {
  const int num_dice = 1;
  const int num_faces = 3;
  SubgameSolvingParams params;
  params.num_iters = 100;
  params.max_depth = 3;
  params.linear_update = true;
  const Game game(num_dice, num_faces);
  auto net = create_zero_net(game.num_hands());

  auto strategy = compute_strategy_recursive_to_leaf(game, params, net);
  auto full_tree = unroll_tree(game);
  ASSERT_EQ(strategy.size(), full_tree.size());
  for (size_t i = 0; i < strategy.size(); ++i) {
    if (game.is_terminal(full_tree[i].state)) continue;
    ASSERT_EQ(strategy[i].size(), game.num_hands())
        << "Bad strategy at node " << i;
    for (int h = 0; h < game.num_hands(); ++h) {
      ASSERT_EQ(strategy[i][h].size(), game.num_actions())
          << "Bad strategy at node " << i << " and hand " << h;
    }
  }
}
