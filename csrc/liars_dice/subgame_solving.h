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
Solvers (FP and CFR) for subgames.
*/

#pragma once

#include <array>
#include <vector>

#include "liars_dice.h"
#include "net_interface.h"
#include "tree.h"

namespace liars_dice {

template <class T>
using Pair = std::array<T, 2>;

// This value is added to all reaches before normalization.
constexpr double kReachSmoothingEps = 1e-80;
// Regrets are clipped at this value instead of zero.
constexpr double kRegretSmoothingEps = 1e-80;

// Indexed by [node, hand, action].
using TreeStrategy = std::vector<std::vector<std::vector<double>>>;

struct TreeStrategyStats;

struct SubgameSolvingParams {
  // Common FP-CFR params.
  int num_iters = 10;
  int max_depth = 2;
  bool linear_update = false;
  bool use_cfr = false;  // Whetehr to use FP or CFR.

  // FP only params.
  bool optimistic = false;

  // CFR-only.
  bool dcfr = false;
  double dcfr_alpha = 0;
  double dcfr_beta = 0;
  double dcfr_gamma = 0;
};

class ISubgameSolver {
 public:
  virtual ~ISubgameSolver() = default;

  // Get values for each hand at the top of the game.
  virtual std::vector<double> get_hand_values(int player_id) const = 0;

  virtual void print_strategy(const std::string& path) const = 0;

  virtual void step(int traverser) = 0;
  // Make params.num_iter steps.
  virtual void multistep() = 0;

  // Matrix of shape [node, hand, action]: responses for every hand and node.
  virtual const TreeStrategy& get_strategy() const = 0;
  // Strategy to use to choose next node in MDP.
  virtual const TreeStrategy& get_sampling_strategy() const {
    return get_strategy();
  }
  // Strategy to use to compute beliefs in a leaf node to create a new subgame
  // in the node.
  virtual const TreeStrategy& get_belief_propogation_strategy() const {
    return get_sampling_strategy();
  }
  // Send current value estimation at the root node to the network.
  virtual void update_value_network() = 0;

  virtual const Tree& get_tree() const = 0;
};

std::vector<float> get_query(const Game& game, int traverser,
                             const PartialPublicState& state,
                             const std::vector<double>& reaches1,
                             const std::vector<double>& reaches2);

std::tuple<int, PartialPublicState, std::vector<double>, std::vector<double>>
deserialize_query(const Game& game, const float* query);

TreeStrategy get_uniform_strategy(const Game& game, const Tree& tree);

void print_strategy(const Game& game, const Tree& tree,
                    const TreeStrategy& strategy, const std::string& fpath);
void print_strategy(const Game& game, const Tree& tree,
                    const TreeStrategy& strategy, std::ostream& stream);
void print_strategy(const Game& game, const Tree& tree,
                    const TreeStrategy& strategy);

// Computes probabilities to win the game for each possible hand assuming that
// the oponents hands are distributed according to beliefs.
std::vector<double> compute_win_probability(const Game& game, Action bet,
                                            const std::vector<double>& beliefs);

inline Pair<std::vector<double>> get_initial_beliefs(const Game& game) {
  Pair<std::vector<double>> beliefs;
  beliefs[0].assign(game.num_hands(), 1.0 / game.num_hands());
  beliefs[1].assign(game.num_hands(), 1.0 / game.num_hands());
  return beliefs;
}

std::unique_ptr<ISubgameSolver> build_solver(
    const Game& game, const PartialPublicState& root,
    const Pair<std::vector<double>>& beliefs,
    const SubgameSolvingParams& params, std::shared_ptr<IValueNet> net);

inline std::unique_ptr<ISubgameSolver> build_solver(
    const Game& game, const SubgameSolvingParams& params,
    std::shared_ptr<IValueNet> net) {
  return build_solver(game, game.get_initial_state(), get_initial_beliefs(game),
                      params, net);
}

inline std::unique_ptr<ISubgameSolver> build_solver(
    const Game& game, const SubgameSolvingParams& params) {
  return build_solver(game, params, /*net=*/nullptr);
}

double compute_exploitability(const Game& game, const TreeStrategy& strategy);
std::array<double, 2> compute_exploitability2(const Game& game,
                                              const TreeStrategy& strategy);

TreeStrategyStats compute_stategy_stats(const Game& game,
                                        const TreeStrategy& strategy);

// Compute EV of the first player for full tree strategies.
// EV(hand) := sum_{z} sum_{op_hand} pi(z|hand, op_hand) U(hand, op_hand | z).
std::vector<double> compute_ev(const Game& game, const TreeStrategy& strategy1,
                               const TreeStrategy& strategy2);
Pair<double> compute_ev2(const Game& game, const TreeStrategy& strategy1,
                         const TreeStrategy& strategy2);

std::vector<std::vector<double>> compute_immediate_regrets(
    const Game& game, const std::vector<TreeStrategy>& strategies);

struct TreeStrategyStats {
  Tree tree;

  // reach_probabilities[p][node][hand] is the probabiliy to get hand `hand` and
  // use to play blueprint to reach node `node`.
  Pair<std::vector<std::vector<double>>> reach_probabilities;

  // values[p][node][hand] is expected value that player `p` can get
  // - the games starts at node node
  // - p has hand `hand`
  // - op hands are defined as noramlized(reach_probabilities[node][1 - p]).
  Pair<std::vector<std::vector<double>>> values;

  // values[p][node] is expected value that player `p` can get
  // - the games starts at node node
  // - p hands are defined as noramlized(reach_probabilities[node][1]).
  // - op hands are defined as noramlized(reach_probabilities[node][1 - p]).
  Pair<std::vector<double>> node_values;

  // Probability to reach a public node if both players play by blueprint.
  std::vector<double> node_reach;
};

}  // namespace liars_dice
