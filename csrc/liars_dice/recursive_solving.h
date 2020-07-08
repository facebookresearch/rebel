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

/*
Recursive training and evaluation.
*/

#pragma once

#include <memory>
#include <random>
#include <vector>

#include "liars_dice.h"
#include "net_interface.h"
#include "subgame_solving.h"

namespace liars_dice {

struct RecursiveSolvingParams {
  int num_dice;
  int num_faces;
  // Probability to explore random action for BR player.
  float random_action_prob = 1.0;
  bool sample_leaf = false;
  SubgameSolvingParams subgame_params;
};

class RlRunner {
 public:
  RlRunner(const RecursiveSolvingParams& params, std::shared_ptr<IValueNet> net,
           int seed)
      : game_(Game(params.num_dice, params.num_faces)),
        subgame_params_(params.subgame_params),
        random_action_prob_(params.random_action_prob),
        sample_leaf_(params.sample_leaf),
        net_(net),
        gen_(seed) {}

  // Deprecated constructor.
  RlRunner(const Game& game, const SubgameSolvingParams& params,
           std::shared_ptr<IValueNet> net, int seed)
      : RlRunner(build_params(game, params), net, seed) {}

  void step();

 private:
  static RecursiveSolvingParams build_params(
      const Game& game, const SubgameSolvingParams& fp_params) {
    RecursiveSolvingParams params;
    params.subgame_params = fp_params;
    params.num_dice = game.num_dice;
    params.num_faces = game.num_faces;
    return params;
  }

  // Samples new state_ from the solver and update beliefs.
  void sample_state(const ISubgameSolver* solver);
  void sample_state_single(const ISubgameSolver* solver);
  void sample_state_to_leaf(const ISubgameSolver* solver);

  // Owning all small resources.
  const Game game_;
  const SubgameSolvingParams subgame_params_;
  const float random_action_prob_;
  const bool sample_leaf_;
  std::shared_ptr<IValueNet> net_;

  // Current state.
  PartialPublicState state_;
  // Buffer to the beliefs.
  Pair<std::vector<double>> beliefs_;

  std::mt19937 gen_;
};

// Compute strategy by recursively solving subgames. Use only the strategy at
// root of the same for the full tree, and proceed to its children.
TreeStrategy compute_strategy_recursive(
    const Game& game, const SubgameSolvingParams& subgame_params,
    std::shared_ptr<IValueNet> net);
// Compute strategy by recursively solving subgames. Use strategy for all
// non-leaf subgame nodes as for full game strategy and proceed with leaf nodes
// in the subgame.
TreeStrategy compute_strategy_recursive_to_leaf(
    const Game& game, const SubgameSolvingParams& subgame_params,
    std::shared_ptr<IValueNet> net);
// Compute strategy by recursively solving subgames in way that mimics training:
// 1. Sample random iteration with linear weigting.
// 2. Copy the sampling strategy for the solver to the full game strategy.
// 3. Compute beliefs in leaves using belief_propogation_strategy start
// recursively.
TreeStrategy compute_sampled_strategy_recursive_to_leaf(
    const Game& game, const SubgameSolvingParams& subgame_params,
    std::shared_ptr<IValueNet> net, int seed, bool root_only = false);

}  // namespace liars_dice