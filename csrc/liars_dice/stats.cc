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

#include "stats.h"

#include <memory>
#include <vector>

#include <torch/torch.h>

#include "liars_dice.h"
#include "net_interface.h"
#include "subgame_solving.h"
#include "util.h"

namespace liars_dice {

namespace {

void compute_depths(const Tree& tree, std::vector<int>* depths, int index = 0,
                    int depth = 0) {
  if (depths->empty()) {
    depths->resize(tree.size());
  }
  depths->at(index) = depth;
  for (auto child : ChildrenIt(tree[index])) {
    compute_depths(tree, depths, child, depth + 1);
  }
}

}  // namespace

float eval_net(const Game& game, const TreeStrategy& net_strategy,
               const TreeStrategy& full_strategy, int mdp_depth, int fp_iters,
               std::shared_ptr<IValueNet> net, bool traverse_by_net,
               bool verbose) {
  const auto full_tree = unroll_tree(game);
  const auto net_stats = compute_stategy_stats(game, net_strategy);
  const auto true_stats = compute_stategy_stats(game, full_strategy);
  if (verbose) {
    if (traverse_by_net) {
      std::cout << "Using net policy to define beliefs\n";
    } else {
      std::cout << "Using FP policy to define beliefs\n";
    }
  }
  const auto traversing_stats = traverse_by_net ? net_stats : true_stats;
  auto node_reach = traversing_stats.node_reach;
  // Get non-terminal nodes at depth mdp_depth and mdp_depth * 2.
  std::vector<int> depths;
  compute_depths(full_tree, &depths);
  std::vector<int> top_node_ids;
  for (size_t i = 0; i < node_reach.size(); ++i) {
    if (depths[i] == mdp_depth || depths[i] == 2 * mdp_depth) {
      if (!game.is_terminal(full_tree[i].state)) {
        top_node_ids.push_back(i);
      }
    }
  }
  // Sort in descending order.
  std::sort(
      top_node_ids.begin(), top_node_ids.end(),
      [&node_reach](int i, int j) { return node_reach[i] > node_reach[j]; });
  const float kMinReach = 1e-6;
  if (verbose) {
    std::cout << "Non-terminal nodes at depth " << mdp_depth << ": "
              << top_node_ids.size() << "\n";
  }
  if (top_node_ids.empty()) {
    std::cout << "Empty list. Exiting.\n";
    return 0.0;
  }
  while (node_reach[top_node_ids.back()] < kMinReach) {
    top_node_ids.pop_back();
  }
  if (verbose) {
    std::cout << "After filtering with reach < " << kMinReach << ": "
              << top_node_ids.size() << "\n";
    std::cout << "Min reach: " << node_reach[top_node_ids.back()] << "\n";
    std::cout << "Max reach: " << node_reach[top_node_ids.front()] << "\n";
  }

  double total_true_reach = 0, total_net_reach = 0;
  for (auto node_id : top_node_ids) {
    total_true_reach += true_stats.node_reach[node_id];
    total_net_reach += net_stats.node_reach[node_id];
  }
  if (verbose) {
    std::cout << "Total reach: true=" << total_true_reach
              << " net=" << total_net_reach << "\n";
  }

  if (top_node_ids.empty()) {
    // that's odd.
    return 0.0;
  }

  std::vector<float> mses;

  for (auto node_id : top_node_ids) {
    Pair<std::vector<double>> beliefs = {
        normalize_probabilities(
            traversing_stats.reach_probabilities[0][node_id]),
        normalize_probabilities(
            traversing_stats.reach_probabilities[1][node_id])};
    const auto& state = full_tree[node_id].state;
    SubgameSolvingParams params;
    params.num_iters = fp_iters;
    params.max_depth = 10000;
    params.linear_update = true;
    auto fp = build_solver(game, state, beliefs, params, nullptr);
    fp->multistep();

    for (int traverser : {0, 1}) {
      auto query = torch::tensor(
          get_query(game, traverser, state, beliefs[0], beliefs[1]));
      auto reach_tensor = torch::tensor(beliefs[traverser]);
      float net_value =
          (net->compute_values(query.unsqueeze(0)).squeeze(0) * reach_tensor)
              .sum()
              .item<float>();
      float br_value =
          (torch::tensor(fp->get_hand_values(traverser)) * reach_tensor)
              .sum()
              .item<float>();
      float blueprint_value = true_stats.node_values[traverser][node_id];

      if (verbose) {
        std::cout << game.state_to_string(state)
                  << "\tnet_reach=" << net_stats.node_reach[node_id]
                  << " true_reach=" << true_stats.node_reach[node_id]
                  << " net_value=" << net_value << " br_value=" << br_value;
        if (!traverse_by_net) std::cout << " blue_value=" << blueprint_value;
        std::cout << "\n";
      }
      mses.push_back(std::pow(net_value - br_value, 2.0));
    }
  }
  float mse = vector_sum(mses) / mses.size();
  if (verbose) std::cout << "Final MSE: " << mse << "\n";
  return mse;
}

}  // namespace liars_dice