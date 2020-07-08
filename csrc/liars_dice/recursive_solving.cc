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

#include "recursive_solving.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include "liars_dice.h"
#include "net_interface.h"
#include "subgame_solving.h"
#include "util.h"

namespace liars_dice {

namespace {

using SubgameSolverBuilder = std::function<std::unique_ptr<ISubgameSolver>(
    const Game& game, int node_id, const PartialPublicState& state,
    const Pair<std::vector<double>>& beliefs)>;

void normalize_beliefs_inplace(std::vector<double>& beliefs) {
  return normalize_probabilities_safe(beliefs, kReachSmoothingEps,
                                      beliefs.data());
}

// Compute strategies for this node and all children.
void compute_strategy_recursive(const Game& game, const Tree& tree, int node_id,
                                const Pair<std::vector<double>>& beliefs,
                                const SubgameSolverBuilder& solver_builder,
                                TreeStrategy* strategy) {
  auto& node = tree[node_id];
  auto& state = node.state;
  if (game.is_terminal(state)) return;

  auto solver = solver_builder(game, node_id, state, beliefs);
  solver->multistep();
  strategy->at(node_id) = solver->get_strategy()[0];

  for (auto child_node_id = node.children_begin;
       child_node_id < node.children_end; ++child_node_id) {
    auto new_beliefs = beliefs;
    auto action =
        child_node_id - node.children_begin + game.get_bid_range(state).first;
    // Update beliefs.
    // P^{t+1}(hand|action) \propto  P^t(action|hand)P^t(hand) .
    for (int hand = 0; hand < game.num_hands(); ++hand) {
      // Assuming that the policy has zeros outside of the range.
      new_beliefs[state.player_id][hand] *= (*strategy)[node_id][hand][action];
    }
    normalize_beliefs_inplace(new_beliefs[state.player_id]);
    compute_strategy_recursive(game, tree, child_node_id, new_beliefs,
                               solver_builder, strategy);
  }
}

void compute_strategy_recursive_to_leaf(
    const Game& game, const Tree& tree, int node_id,
    const Pair<std::vector<double>>& beliefs,
    const SubgameSolverBuilder& solver_builder, bool use_samplig_strategy,
    TreeStrategy* strategy) {
  auto& node = tree[node_id];
  auto& state = node.state;
  if (game.is_terminal(state)) return;

  auto solver = solver_builder(game, node_id, state, beliefs);
  solver->multistep();

  // Tree traversal queue storing tuples:
  //   (full_node_id, partial_node_id, unnormalized beliefs at the node).
  // We do BFS traversal. For each node:
  // - copy the policy from the partial (solver) tree to strategy.
  // - add children to the queue with propoer believes.
  // - for non-termial leaves of the solver tree, do a recursive call.
  std::deque<std::tuple<int, int, Pair<std::vector<double>>>> traversal_queue;
  traversal_queue.emplace_back(node_id, 0, beliefs);

  const TreeStrategy& partial_strategy = use_samplig_strategy
                                             ? solver->get_sampling_strategy()
                                             : solver->get_strategy();
  const TreeStrategy& partial_belief_strategy =
      use_samplig_strategy ? solver->get_belief_propogation_strategy()
                           : solver->get_strategy();
  const Tree& partial_tree = solver->get_tree();
  while (!traversal_queue.empty()) {
    auto [full_node_id, partial_node_id, node_reaches] =
        std::move(traversal_queue.front());
    traversal_queue.pop_front();
    (*strategy)[full_node_id] = partial_strategy[partial_node_id];
    const auto& full_node = tree[full_node_id];
    const auto& partial_node = partial_tree[partial_node_id];
    assert(partial_node.num_children() == 0 ||
           partial_node.num_children() == full_node.num_children());
    assert(partial_node.state == full_node.state);
    for (int i = 0; i < partial_node.num_children(); ++i) {
      auto child_reaches = node_reaches;
      const int pid = full_node.state.player_id;
      const int action = game.get_bid_range(full_node.state).first + i;
      for (int hand = 0; hand < game.num_hands(); ++hand) {
        child_reaches[pid][hand] *=
            partial_belief_strategy[partial_node_id][hand][action];
      }
      traversal_queue.emplace_back(full_node.children_begin + i,
                                   partial_node.children_begin + i,
                                   child_reaches);
    }
    if (partial_node.num_children() == 0 && full_node.num_children() != 0) {
      normalize_beliefs_inplace(node_reaches[0]);
      normalize_beliefs_inplace(node_reaches[1]);
      compute_strategy_recursive_to_leaf(game, tree, full_node_id, node_reaches,
                                         solver_builder, use_samplig_strategy,
                                         strategy);
    }
  }
}

TreeStrategy compute_strategy_with_solver(
    const Game& game, const SubgameSolverBuilder& solver_builder) {
  const Tree tree = unroll_tree(game);
  TreeStrategy strategy(tree.size());
  const auto beliefs = get_initial_beliefs(game);
  compute_strategy_recursive(game, tree, /*node_id=*/0, beliefs, solver_builder,
                             &strategy);
  return strategy;
}

TreeStrategy compute_strategy_with_solver_to_leaf(
    const Game& game, const SubgameSolverBuilder& solver_builder,
    bool use_samplig_strategy = false) {
  const Tree tree = unroll_tree(game);
  TreeStrategy strategy(tree.size());
  const auto beliefs = get_initial_beliefs(game);
  compute_strategy_recursive_to_leaf(game, tree, /*node_id=*/0, beliefs,
                                     solver_builder, use_samplig_strategy,
                                     &strategy);
  return strategy;
}

}  // namespace

void RlRunner::step() {
  state_ = game_.get_initial_state();
  beliefs_[0].assign(game_.num_hands(), 1.0 / game_.num_hands());
  beliefs_[1].assign(game_.num_hands(), 1.0 / game_.num_hands());
  // std::cout << "state: " << game_.state_to_string(state_) << "\n";
  while (!game_.is_terminal(state_)) {
    auto solver = build_solver(game_, state_, beliefs_, subgame_params_, net_);

    const int act_iteration =
        std::uniform_int_distribution<>(0, subgame_params_.num_iters)(gen_);
    for (int iter = 0; iter < act_iteration; ++iter) {
      solver->step(/*traverser=*/iter % 2);
    }
    // Sample a new state to explore.
    sample_state(solver.get());
    for (int iter = act_iteration; iter < subgame_params_.num_iters; ++iter) {
      solver->step(/*traverser=*/iter % 2);
    }

    // Collect the values at the top of the tree.
    solver->update_value_network();
  }
}

void RlRunner::sample_state(const ISubgameSolver* solver) {
  if (sample_leaf_) {
    sample_state_to_leaf(solver);
  } else {
    sample_state_single(solver);
  }
}

void RlRunner::sample_state_to_leaf(const ISubgameSolver* solver) {
  const auto& tree = solver->get_tree();
  // List of (node, action) pairs.
  std::vector<std::pair<int, Action>> path;
  {
    int node_id = 0;
    const auto br_sampler = std::uniform_int_distribution<>(0, 1)(gen_);
    const auto& strategy = solver->get_sampling_strategy();
    auto sampling_beliefs = beliefs_;
    while (tree[node_id].num_children()) {
      const auto eps = std::uniform_real_distribution<float>(0, 1)(gen_);
      Action action;
      const auto& state = tree[node_id].state;
      const auto [action_begin, action_end] = game_.get_bid_range(state);
      if (state.player_id == br_sampler && eps < random_action_prob_) {
        std::uniform_int_distribution<> dis(action_begin, action_end - 1);
        action = dis(gen_);
      } else {
        const auto& beliefs = sampling_beliefs[state.player_id];
        std::discrete_distribution<> dis(beliefs.begin(), beliefs.end());
        const int hand = dis(gen_);
        const std::vector<double>& policy = strategy[node_id][hand];
        std::discrete_distribution<> action_dis(policy.begin(), policy.end());
        action = action_dis(gen_);
        assert(action >= action_begin && action < action_end);
      }
      // Update beliefs.
      // Policy[hand, action] := P(action | hand).
      const auto& policy = strategy[node_id];
      // P^{t+1}(hand|action) \propto  P^t(action|hand)P^t(hand) .
      for (int hand = 0; hand < game_.num_hands(); ++hand) {
        // Assuming that the policy has zeros outside of the range.
        sampling_beliefs[state.player_id][hand] *= policy[hand][action];
      }
      normalize_beliefs_inplace(sampling_beliefs[state.player_id]);
      path.emplace_back(node_id, action);
      node_id = tree[node_id].children_begin + action - action_begin;
    }
  }

  // We do another pass over the path to compute beliefs accroding to
  // `get_belief_propogation_strategy` that could differ from the sampling
  // strategy.
  for (auto [node_id, action] : path) {
    const auto action_begin = game_.get_bid_range(state_).first;
    const auto& policy = solver->get_belief_propogation_strategy()[node_id];
    for (int hand = 0; hand < game_.num_hands(); ++hand) {
      // Assuming that the policy has zeros outside of the range.
      beliefs_[state_.player_id][hand] *= policy[hand][action];
    }
    normalize_beliefs_inplace(beliefs_[state_.player_id]);
    int child_node_id = tree[node_id].children_begin + action - action_begin;
    state_ = tree[child_node_id].state;
  }
}

void RlRunner::sample_state_single(const ISubgameSolver* solver) {
  Action action;
  const auto br_sampler = std::uniform_int_distribution<>(0, 1)(gen_);
  const auto eps = std::uniform_real_distribution<float>(0, 1)(gen_);
  if (state_.player_id == br_sampler && eps < random_action_prob_) {
    auto [action_begin, action_end] = game_.get_bid_range(state_);
    std::uniform_int_distribution<> dis(action_begin, action_end - 1);
    action = dis(gen_);
  } else {
    const auto& beliefs = beliefs_[state_.player_id];
    std::discrete_distribution<> dis(beliefs.begin(), beliefs.end());
    const int hand = dis(gen_);
    const std::vector<double>& policy =
        solver->get_sampling_strategy()[0][hand];
    std::discrete_distribution<> action_dis(policy.begin(), policy.end());
    action = action_dis(gen_);
  }
  // Update beliefs.
  // Policy[hand, action] := P(action | hand).
  const auto& policy = solver->get_belief_propogation_strategy()[0];
  // P^{t+1}(hand|action) \propto  P^t(action|hand)P^t(hand) .
  for (int hand = 0; hand < game_.num_hands(); ++hand) {
    // Assuming that the policy has zeros outside of the range.
    beliefs_[state_.player_id][hand] *= policy[hand][action];
  }
  normalize_beliefs_inplace(beliefs_[state_.player_id]);
  state_ = game_.act(state_, action);
}

TreeStrategy compute_strategy_recursive(
    const Game& game, const SubgameSolvingParams& subgame_params,
    std::shared_ptr<IValueNet> net) {
  SubgameSolverBuilder solver_builder =
      [net, subgame_params](const Game& game, int /*node_id*/,
                            const PartialPublicState& state,
                            const Pair<std::vector<double>>& beliefs) {
        return build_solver(game, state, beliefs, subgame_params, net);
      };
  return compute_strategy_with_solver(game, solver_builder);
}

TreeStrategy compute_strategy_recursive_to_leaf(
    const Game& game, const SubgameSolvingParams& subgame_params,
    std::shared_ptr<IValueNet> net) {
  SubgameSolverBuilder solver_builder =
      [net, subgame_params](const Game& game, int /*node_id*/,
                            const PartialPublicState& state,
                            const Pair<std::vector<double>>& beliefs) {
        return build_solver(game, state, beliefs, subgame_params, net);
      };
  return compute_strategy_with_solver_to_leaf(game, solver_builder);
}

TreeStrategy compute_sampled_strategy_recursive_to_leaf(
    const Game& game, const SubgameSolvingParams& subgame_params,
    std::shared_ptr<IValueNet> net, int seed, bool root_only) {
  std::mt19937 gen(seed);
  // Emulate linear weigting: choose only even iterations.
  std::vector<double> iteration_weights;
  for (int i = 0; i < subgame_params.num_iters; ++i) {
    iteration_weights.push_back(i % 2 ? 0.0 : (i / 2. + 1));
  }

  SubgameSolverBuilder solver_builder =
      [net, subgame_params, iteration_weights, root_only, &gen](
          const Game& game, int node_id, const PartialPublicState& state,
          const Pair<std::vector<double>>& beliefs) {
        std::discrete_distribution<int> iteration_distribution(
            iteration_weights.begin(), iteration_weights.end());
        const int act_iteration = iteration_distribution(gen);
        auto params = subgame_params;
        params.num_iters = act_iteration;
        if (root_only && node_id != 0) {
          params.max_depth = 100000;
        }
        return build_solver(game, state, beliefs, params, net);
      };
  return compute_strategy_with_solver_to_leaf(game, solver_builder,
                                              /*use_samplig_strategy=*/true);
}

}  // namespace liars_dice
