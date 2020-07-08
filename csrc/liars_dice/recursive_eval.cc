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

#include <algorithm>
#include <iostream>
#include <map>
#include <string>

#include "real_net.h"
#include "recursive_solving.h"
#include "stats.h"
#include "subgame_solving.h"
#include "util.h"

using namespace liars_dice;

void report_regrets(const Game& game,
                    const std::vector<TreeStrategy>& strategy_list,
                    bool print_regret, bool print_regret_summary, int depth) {
  auto full_tree = unroll_tree(game);
  auto regrets = compute_immediate_regrets(game, strategy_list);
  if (print_regret) {
    std::cout << "\tRegrets: ";
    for (int node = 0; node < 20; ++node) {
      for (auto i : regrets[node]) std::cout << i << " ";
      std::cout << "| ";
    }
    std::cout << "\n";
  }
  if (print_regret_summary) {
    double top_regret = 0, bottom_regret = 0;
    for (size_t node_id = 0; node_id < full_tree.size(); ++node_id) {
      if (full_tree[node_id].depth < depth) {
        top_regret += vector_sum(regrets[node_id]);
      } else {
        bottom_regret += vector_sum(regrets[node_id]);
      }
    }
    std::cout << "\tRegrets (depth<=" << depth << ")/rest: " << top_regret
              << "/" << bottom_regret;
  }
}

void report_game_stats(const Game& game, const TreeStrategy& strategy) {
  const auto full_tree = unroll_tree(game);
  auto stats = compute_stategy_stats(game, strategy);
  std::cout << "Some stats on reach and values if played by blueprint\n";
  std::cout << "node\tstate\treach\tvalues p0\tvalues p1\n";
  for (size_t i = 0; i < std::min<size_t>(20, stats.node_reach.size()); ++i) {
    std::cout << std::setw(5) << i << " "
              << game.state_to_string_short(full_tree[i].state) << "\t"
              << std::setprecision(4) << stats.node_reach[i];
    for (auto p : {0, 1}) {
      std::cout << "\t";
      std::cout << std::setw(7) << std::setprecision(4)
                << stats.node_values[p][i] << " = ";
      for (auto value : stats.values[p][i]) {
        std::cout << std::setw(7) << std::setprecision(4) << value << " ";
      }
    }
    std::cout << "\n";
  }
}

int get_depth(const Tree& tree, int root = 0) {
  int depth = 1;
  for (auto child : ChildrenIt(tree[root])) {
    depth = std::max(depth, 1 + get_depth(tree, child));
  }
  return depth;
}
torch::Tensor tree_strategy_to_tensor(const TreeStrategy& strategy) {
  torch::Tensor tensor =
      torch::zeros({(int64_t)strategy.size(), (int64_t)strategy[0].size(),
                    (int64_t)strategy[0][0].size()});
  auto acc = tensor.accessor<float, 3>();

  for (size_t node = 0; node < strategy.size(); ++node) {
    for (size_t hand = 0; hand < strategy[node].size(); ++hand) {
      for (size_t action = 0; action < strategy[node][hand].size(); ++action) {
        acc[node][hand][action] = strategy[node][hand][action];
      }
    }
  }
  return tensor;
}

TreeStrategy tensor_to_tree_strategy(const torch::Tensor tensor) {
  TreeStrategy strategy(tensor.size(0));
  const int num_hands = tensor.size(1);
  const int num_values = tensor.size(2);
  auto acc = tensor.accessor<float, 3>();

  for (size_t node = 0; node < strategy.size(); ++node) {
    strategy[node].resize(num_hands);
    for (size_t hand = 0; hand < strategy[node].size(); ++hand) {
      strategy[node][hand].resize(num_values);
      for (size_t action = 0; action < strategy[node][hand].size(); ++action) {
        strategy[node][hand][action] = acc[node][hand][action];
      }
    }
  }
  return strategy;
}

struct ParallerSampledStrategyComputor {
  ParallerSampledStrategyComputor(
      const Game& game, std::function<std::shared_ptr<IValueNet>()> net_builder,
      int num_repeats, SubgameSolvingParams base_params, int mdp_depth,
      int num_threads, bool root_only) {
    SubgameSolvingParams params = base_params;
    params.max_depth = mdp_depth;

    auto worker = [this, params, net_builder, game, root_only]() {
      auto net = net_builder();
      const auto tree = unroll_tree(game);
      while (true) {
        int strategy_id;
        {
          std::lock_guard<std::mutex> lk(mutex);
          if (tasks.empty()) break;
          strategy_id = tasks.back();
          tasks.pop_back();
        }
        auto sampled_strategy = compute_sampled_strategy_recursive_to_leaf(
            game, params, net, /*seed=*/strategy_id, root_only);
        auto sampled_strategy_tensor =
            tree_strategy_to_tensor(sampled_strategy);
        const auto stats = compute_stategy_stats(game, sampled_strategy);
        // Weigting infoset (node, hand) by probability to get the hand and
        // reach the infoset by player(node).
        std::vector<float> reaches;
        for (size_t node = 0; node < tree.size(); ++node) {
          const auto& node_reaches =
              stats.reach_probabilities[tree[node].state.player_id][node];
          for (auto v : node_reaches) reaches.push_back(v);
        }
        auto node_reach_tensor =
            torch::tensor(reaches).view({(int64_t)tree.size(), -1, 1}).clone();

        {
          std::lock_guard<std::mutex> lk(mutex);
          is_done[strategy_id] = 1;
          results[strategy_id] =
              std::make_pair(sampled_strategy_tensor, node_reach_tensor);
        }
      }
    };

    for (int i = num_repeats; i-- > 0;) tasks.push_back(i);
    is_done.resize(num_repeats);
    results.resize(num_repeats);
    for (int i = 0; i < num_threads; ++i)
      threads.push_back(std::thread(worker));
  }

  ~ParallerSampledStrategyComputor() {
    {
      std::lock_guard<std::mutex> lk(mutex);
      tasks.clear();
    }
    for (auto& t : threads) t.join();
  }

  std::pair<torch::Tensor, torch::Tensor> get(int strategy_index) {
    while (true) {
      {
        std::lock_guard<std::mutex> lk(mutex);
        if (is_done[strategy_index]) return results[strategy_index];
      }
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }

  std::vector<std::thread> threads;
  std::vector<int> tasks;
  std::vector<int> is_done;
  std::vector<std::pair<torch::Tensor, torch::Tensor>> results;
  std::mutex mutex;
};

int main(int argc, char* argv[]) {
  int num_dice = 1;
  int num_faces = 4;
  int subgame_iters = 1024;
  int mdp_depth = -1;
  int num_repeats = -1;
  std::string net_path;
  bool repeat_oracle_net = false;
  bool no_linear = false;
  bool root_only = false;
  bool print_regret = false;
  bool print_regret_summary = false;
  int eval_oracle_values_iters = -1;
  int num_threads = 10;
  SubgameSolvingParams base_params;
  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  {
    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];
      if (arg == "--num_dice") {
        assert(i + 1 < argc);
        num_dice = std::stoi(argv[++i]);
      } else if (arg == "--num_faces") {
        assert(i + 1 < argc);
        num_faces = std::stoi(argv[++i]);
      } else if (arg == "--subgame_iters") {
        assert(i + 1 < argc);
        subgame_iters = std::stoi(argv[++i]);
      } else if (arg == "--mdp_depth") {
        assert(i + 1 < argc);
        mdp_depth = std::stoi(argv[++i]);
      } else if (arg == "--num_threads") {
        assert(i + 1 < argc);
        num_threads = std::stoi(argv[++i]);
      } else if (arg == "--num_repeats") {
        assert(i + 1 < argc);
        num_repeats = std::stoi(argv[++i]);
      } else if (arg == "--root_only") {
        root_only = true;
      } else if (arg == "--repeat_oracle_net") {
        repeat_oracle_net = true;
      } else if (arg == "--net") {
        assert(i + 1 < argc);
        net_path = argv[++i];
      } else if (arg == "--print_regret") {
        print_regret = true;
      } else if (arg == "--print_regret_summary") {
        print_regret_summary = true;
      } else if (arg == "--no_linear") {
        no_linear = true;
      } else if (arg == "--optimistic") {
        base_params.optimistic = true;
      } else if (arg == "--eval_oracle_values_iters") {
        assert(i + 1 < argc);
        eval_oracle_values_iters = std::stoi(argv[++i]);
      } else if (arg == "--cfr") {
        base_params.use_cfr = true;
      } else if (arg == "--dcfr") {
        base_params.dcfr = true;
        base_params.dcfr_alpha = std::atof(argv[++i]);
        base_params.dcfr_beta = std::atof(argv[++i]);
        base_params.dcfr_gamma = std::atof(argv[++i]);
      } else {
        std::cerr << "Unknown flag: " << arg << "\n";
        return -1;
      }
    }
  }
  assert(num_dice != -1);
  assert(num_faces != -1);

  const Game game(num_dice, num_faces);
  std::cout << "num_dice=" << num_dice << " num_faces=" << num_faces << "\n";
  const auto full_tree = unroll_tree(game);
  std::cout << "Tree of depth " << get_depth(full_tree) << " has "
            << full_tree.size() << " nodes\n";

  std::cout << "##############################################\n";
  std::cout << "##### Solving the game for the full tree #####\n";
  std::cout << "##############################################\n";
  TreeStrategy full_strategy;
  base_params.num_iters = subgame_iters;
  base_params.linear_update = !no_linear && !base_params.dcfr;
  {
    SubgameSolvingParams params = base_params;
    params.max_depth = 100000;
    auto fp = build_solver(game, params);

    std::vector<TreeStrategy> strategy_list;

    for (int iter = 0; iter < subgame_iters; ++iter) {
      fp->step(iter % 2);
      if (iter % 2 == 0 && params.use_cfr) {
        strategy_list.push_back(fp->get_sampling_strategy());
      }
      if (((iter + 1) & iter) == 0 || iter + 1 == subgame_iters) {
        auto values = compute_exploitability2(game, fp->get_strategy());
        printf("Iter=%8d exploitabilities=(%.3e, %.3e) sum=%.3e\n", iter + 1,
               values[0], values[1], (values[0] + values[1]) / 2.);
      }
    }

    full_strategy = fp->get_strategy();
    auto explotabilities = compute_exploitability2(game, full_strategy);
    std::cout << "Full FP exploitability: "
              << (explotabilities[0] + explotabilities[1]) / 2. << " ("
              << explotabilities[0] << "," << explotabilities[1] << ")"
              << std::endl;
    // report_game_stats(game, fp->get_strategy());
    if (!strategy_list.empty()) {
      report_regrets(game, strategy_list, print_regret, print_regret_summary,
                     mdp_depth);
      std::cout << "\n";
    }
    print_strategy(game, fp->get_tree(), fp->get_strategy(),
                   "strategy.full.txt");
  }
  std::vector<std::pair<std::string, TreeStrategy>> all_strategies;
  all_strategies.emplace_back("full_tree", full_strategy);
  if (!net_path.empty()) {
    assert(mdp_depth > 0);
    std::shared_ptr<IValueNet> net =
        net_path == "zero"
            ? liars_dice::create_zero_net(game.num_hands(), false)
            : liars_dice::create_torchscript_net(net_path);

    std::cout << "##############################################\n";
    std::cout << "##### Recursive solving                      #\n";
    std::cout << "##############################################\n";
    if (num_repeats > 0) {
      torch::Tensor summed_stategy, summed_reach;
      TreeStrategy final_strategy;

      auto net_builder = [=]() {
        if (repeat_oracle_net) {
          SubgameSolvingParams oracle_net_params = base_params;
          oracle_net_params.max_depth = 100000;
          if (eval_oracle_values_iters > 0) {
            oracle_net_params.num_iters = eval_oracle_values_iters;
          }
          return liars_dice::create_oracle_value_predictor(game,
                                                           oracle_net_params);
        } else {
          return liars_dice::create_torchscript_net(net_path, "cpu");
        }
      };

      ParallerSampledStrategyComputor computer(game, net_builder, num_repeats,
                                               base_params, mdp_depth,
                                               num_threads, root_only);
      std::vector<TreeStrategy> strategy_list;
      for (int strategy_id = 0; strategy_id < num_repeats; ++strategy_id) {
        {
          auto [sampled_strategy_tensor, node_reach_tensor] =
              computer.get(strategy_id);

          if (strategy_id == 0) {
            summed_stategy = sampled_strategy_tensor * node_reach_tensor;
            summed_reach = node_reach_tensor;
          } else {
            summed_stategy += sampled_strategy_tensor * node_reach_tensor;
            summed_reach += node_reach_tensor;
          }
          if (base_params.use_cfr) {
            strategy_list.push_back(
                tensor_to_tree_strategy(sampled_strategy_tensor));
          }
        }

        final_strategy =
            tensor_to_tree_strategy(summed_stategy / (summed_reach + 1e-6));
        if (((strategy_id + 1) & strategy_id) == 0 ||
            strategy_id + 1 == num_repeats) {
          std::cout << std::setw(5) << strategy_id + 1 << ": ";
          auto explotabilities = compute_exploitability2(game, final_strategy);
          auto evs = compute_ev2(game, full_strategy, final_strategy);
          std::cout << (explotabilities[0] + explotabilities[1]) / 2. << " ("
                    << explotabilities[0] << "," << explotabilities[1] << ")"
                    << "\tEV of full: ";
          std::cout << (evs[0] + evs[1]) / 2 << " (" << evs[0] << "," << evs[1]
                    << ")";
          if (!strategy_list.empty()) {
            report_regrets(game, strategy_list, print_regret,
                           print_regret_summary, mdp_depth);
          }
          std::cout << std::endl;
          const auto name = (repeat_oracle_net ? "repeated oracle toleaf "
                                               : "repeated toleaf ") +
                            std::to_string(strategy_id + 1);
          all_strategies.emplace_back(name, final_strategy);
          print_strategy(game, full_tree, final_strategy,
                         "strategy.repeated.txt");
        }
      }
    }
  }
  // Reporting in human-readable format.
  std::vector<std::pair<std::string, std::string>> result;
  std::vector<std::pair<std::string, std::string>> result_ev;
  result.emplace_back("net", net_path);
  result_ev.emplace_back("net", net_path);
  for (auto [name, mdp_strategy] : all_strategies) {
    std::cout << " " << name << " ";
    assert(mdp_strategy.size() == full_tree.size());
    auto explotabilities = compute_exploitability2(game, mdp_strategy);
    auto evs = compute_ev2(game, full_strategy, mdp_strategy);
    std::cout << (explotabilities[0] + explotabilities[1]) / 2. << " ("
              << explotabilities[0] << "," << explotabilities[1] << ")"
              << "\n\tEV of full: ";
    std::cout << (evs[0] + evs[1]) / 2. << " (" << evs[0] << "," << evs[1]
              << ")";
    std::cout << std::endl;
    result.emplace_back(
        name, std::to_string((explotabilities[0] + explotabilities[1]) / 2.));
    result_ev.emplace_back(name, std::to_string((evs[0] + evs[1]) / 2.));
  }
  // Reporting as JSON.
  std::vector<
      std::tuple<std::string, std::vector<std::pair<std::string, std::string>>>>
      all_results = {{"XXX", result}, {"YYY", result_ev}};
  for (auto [tag, dict] : all_results) {
    std::cout << tag << " {";
    bool first = true;
    for (auto [k, v] : dict) {
      if (first) {
        first = false;
      } else {
        std::cout << ", ";
      }
      std::cout << '"' << k << "\":\"" << v << "\"";
    }
    std::cout << "}" << std::endl;
  }
}
