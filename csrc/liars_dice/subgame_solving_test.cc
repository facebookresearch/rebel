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
#include "subgame_solving.h"
#include "util.h"

using namespace liars_dice;

namespace {
double compute_fp_exploitability(const Game& game,
                                 const PartialPublicState& root,
                                 const Pair<std::vector<double>>& beliefs,
                                 const SubgameSolvingParams& params,
                                 std::shared_ptr<IValueNet> net) {
  assert(beliefs[0].size() == static_cast<size_t>(game.num_hands()));
  assert(beliefs[1].size() == static_cast<size_t>(game.num_hands()));
  auto solver = build_solver(game, root, beliefs, params, net);
  solver->multistep();
  std::array<double, 2> values =
      compute_exploitability2(game, solver->get_strategy());
  return (values[0] + values[1]) / 2.;
}
// No-network version assumes that all leaf nodes are final.
double compute_fp_exploitability(const Game& game,
                                 const PartialPublicState& root,
                                 const Pair<std::vector<double>>& beliefs,
                                 const SubgameSolvingParams& params) {
  return compute_fp_exploitability(game, root, beliefs, params, nullptr);
}
}  // namespace

TEST(EvaluateTerninalNode, TestSingleDiceOneHotBelief) {
  const int num_dice = 1;
  const int num_faces = 6;
  Game game(num_dice, num_faces);

  std::vector<double> beliefs(game.num_hands());
  ASSERT_EQ(game.num_hands(), 6);
  for (int ophand = 0; ophand < game.num_hands(); ++ophand) {
    beliefs.assign(game.num_hands(), 0.0);
    beliefs[ophand] = 1;
    for (Action bet = 0; bet < game.num_actions() - 1; ++bet) {
      const UnpackedAction unpacked_bet = game.unpack_action(bet);
      auto values = compute_win_probability(game, bet, beliefs);
      for (int myhand = 0; myhand < game.num_hands(); ++myhand) {
        // std::cerr << "myhand=" << myhand << " ophand=" << ophand
        //           << " bet=" << game.action_to_string(bet) << std::endl;
        int matches = 0;
        if (myhand == unpacked_bet.face || myhand == 5) ++matches;
        if (ophand == unpacked_bet.face || ophand == 5) ++matches;
        const double true_value = matches >= unpacked_bet.quantity ? 1.0 : 0.0;
        ASSERT_DOUBLE_EQ(true_value, values[myhand]);
      }
    }
  }
}

TEST(EvaluateTerninalNode, TestTwoDiceOneHotBelief) {
  const int num_dice = 2;
  const int num_faces = 3;
  Game game(num_dice, num_faces);

  std::vector<double> beliefs(game.num_hands());
  ASSERT_EQ(game.num_hands(), 3 * 3);
  for (int ophand = 0; ophand < game.num_hands(); ++ophand) {
    beliefs.assign(game.num_hands(), 0.0);
    beliefs[ophand] = 1;
    for (Action bet = 0; bet < game.num_actions() - 1; ++bet) {
      const UnpackedAction unpacked_bet = game.unpack_action(bet);
      auto values = compute_win_probability(game, bet, beliefs);
      for (int myhand = 0; myhand < game.num_hands(); ++myhand) {
        // std::cerr << "myhand=" << myhand << " ophand=" << ophand
        //           << " bet=" << game.action_to_string(bet) << std::endl;
        int matches = 0;
        if (myhand % num_faces == unpacked_bet.face || myhand % num_faces == 2)
          ++matches;
        if (ophand % num_faces == unpacked_bet.face || ophand % num_faces == 2)
          ++matches;
        if (myhand / num_faces == unpacked_bet.face || myhand / num_faces == 2)
          ++matches;
        if (ophand / num_faces == unpacked_bet.face || ophand / num_faces == 2)
          ++matches;
        const double true_value = matches >= unpacked_bet.quantity ? 1.0 : 0.0;
        ASSERT_DOUBLE_EQ(true_value, values[myhand]);
      }
    }
  }
}

TEST(FictiousTest, TestOneDiceOneFace) {
  const int num_dice = 1;
  const int num_faces = 1;
  SubgameSolvingParams params;
  params.num_iters = 3500;
  params.max_depth = 100;
  Game game(num_dice, num_faces);

  const auto root = game.get_initial_state();

  const auto initial_beliefs = get_initial_beliefs(game);
  ASSERT_EQ(game.num_hands(), 1);
  ASSERT_EQ(initial_beliefs[0][0], 1.0);
  const auto value =
      compute_fp_exploitability(game, root, initial_beliefs, params);
  ASSERT_GE(value, 0.0);
  ASSERT_LT(value, 1e-3);
}

TEST(FictiousTest, TestOneDiceOneFaceLinear) {
  const int num_dice = 1;
  const int num_faces = 1;
  SubgameSolvingParams params;
  params.num_iters = 3500;
  params.max_depth = 100;
  params.linear_update = true;
  Game game(num_dice, num_faces);

  const auto root = game.get_initial_state();

  const auto initial_beliefs = get_initial_beliefs(game);
  ASSERT_EQ(game.num_hands(), 1);
  ASSERT_EQ(initial_beliefs[0][0], 1.0);
  const auto value =
      compute_fp_exploitability(game, root, initial_beliefs, params);
  ASSERT_GE(value, 0.0);
  ASSERT_LT(value, 1e-3);
}

TEST(FictiousTest, TestOneDiceTwoFaces) {
  const int num_dice = 1;
  const int num_faces = 2;
  SubgameSolvingParams params;
  params.num_iters = 10000;
  params.max_depth = 1000;
  Game game(num_dice, num_faces);

  const auto root = game.get_initial_state();

  const auto initial_beliefs = get_initial_beliefs(game);
  const auto value =
      compute_fp_exploitability(game, root, initial_beliefs, params);
  ASSERT_GE(value, 0.0);
  ASSERT_LT(value, 1e-3);
}

TEST(CFRTest, TestOneDiceTwoFacesCfr) {
  const int num_dice = 1;
  const int num_faces = 2;
  SubgameSolvingParams params;
  params.num_iters = 180;
  params.max_depth = 1000;
  params.linear_update = true;
  params.use_cfr = true;
  Game game(num_dice, num_faces);

  const auto root = game.get_initial_state();

  const auto initial_beliefs = get_initial_beliefs(game);
  const auto value =
      compute_fp_exploitability(game, root, initial_beliefs, params);
  ASSERT_GE(value, 0.0);
  ASSERT_LT(value, 1e-3);
}

TEST(CFRTest, TestOneDiceTwoFacesCfrRegrets) {
  const int num_dice = 1;
  const int num_faces = 2;

  SubgameSolvingParams params;
  params.num_iters = 4000;
  params.max_depth = 1000;
  params.use_cfr = true;
  params.linear_update = false;
  Game game(num_dice, num_faces);

  auto solver = build_solver(game, params);
  std::vector<TreeStrategy> strategies;
  for (int i = 0; i < params.num_iters; ++i) {
    if (i % 2 == 0) {
      strategies.push_back(solver->get_sampling_strategy());
    }
    solver->step(i % 2);
  }

  const auto regrets = compute_immediate_regrets(game, strategies);
  for (size_t node = 0; node < regrets.size(); ++node) {
    for (int hand = 0; hand < game.num_hands(); ++hand) {
      ASSERT_LE(regrets[node][hand], 1e-2)
          << "node=" << node << " hand=" << hand;
    }
  }
}

TEST(FictiousTest, TestOneDiceThreeFacesLinear) {
  const int num_dice = 1;
  const int num_faces = 3;
  SubgameSolvingParams params;
  params.num_iters = 1 << 12;
  params.max_depth = 1000;
  params.linear_update = true;
  Game game(num_dice, num_faces);

  const auto root = game.get_initial_state();
  const auto initial_beliefs = get_initial_beliefs(game);
  const auto value =
      compute_fp_exploitability(game, root, initial_beliefs, params);
  ASSERT_GE(value, 0.0);
  ASSERT_LT(value, 2e-3);
}

TEST(FictiousTest, TestOneDiceThreeFacesLinearEV) {
  const int num_dice = 1;
  const int num_faces = 3;
  SubgameSolvingParams params;
  params.num_iters = 1 << 12;
  params.max_depth = 1000;
  params.linear_update = true;
  Game game(num_dice, num_faces);

  auto solver = build_solver(game, params);
  solver->multistep();
  const auto values =
      compute_ev2(game, solver->get_strategy(), solver->get_strategy());
  std::cout << values[0] << " " << values[1] << "\n";
  ASSERT_LE(values[0], 2.);
  ASSERT_GE(values[0], -2.);
  ASSERT_NEAR(values[0] + values[1], 0., 1e-6);
}

TEST(FictiousTest, TestOneDiceFourFacesOracleNet) {
  const int num_dice = 1;
  const int num_faces = 3;
  SubgameSolvingParams params;
  params.num_iters = 1 << 12;
  params.max_depth = 5;
  params.linear_update = true;
  Game game(num_dice, num_faces);

  SubgameSolvingParams oracle_net_params = params;
  params.max_depth = 50;

  const auto root = game.get_initial_state();
  const auto initial_beliefs = get_initial_beliefs(game);
  auto net = create_oracle_value_predictor(game, oracle_net_params);
  const auto value =
      compute_fp_exploitability(game, root, initial_beliefs, params, net);
  ASSERT_GE(value, 0.0);
  ASSERT_LT(value, 2e-3);
}

TEST(QueryTest, TestQueryDeserialization) {
  const int num_dice = 1;
  const int num_faces = 3;
  Game game(num_dice, num_faces);
  const auto tree = unroll_tree(game);
  std::vector<double> beliefs1, beliefs2;
  for (int i = 0; i < game.num_hands(); ++i) beliefs1.push_back(i);
  for (int i = 0; i < game.num_hands(); ++i) beliefs2.push_back(i + 0.5);
  normalize_probabilities(beliefs1, &beliefs1);
  normalize_probabilities(beliefs2, &beliefs2);
  for (int traverser : {0, 1}) {
    for (const auto& node : tree) {
      const auto& state = node.state;
      // Value net cannot be queries in terminal nodes.
      if (game.is_terminal(state)) continue;
      const auto query = get_query(game, traverser, state, beliefs1, beliefs2);

      const auto [deserialized_traverser, deserialized_state,
                  deserialized_beliefs1, deserialized_beliefs2] =
          deserialize_query(game, query.data());
      ASSERT_EQ(state.player_id, deserialized_state.player_id);
      ASSERT_EQ(traverser, deserialized_traverser);
      ASSERT_EQ(state.last_bid, deserialized_state.last_bid);
      for (int i = 0; i < game.num_hands(); ++i) {
        ASSERT_NEAR(beliefs1[i], deserialized_beliefs1[i], 1e-6);
        ASSERT_NEAR(beliefs2[i], deserialized_beliefs2[i], 1e-6);
      }
    }
  }
}

TEST(UtilsTest, TestProbNormalization) {
  std::vector<double> probs{2.93185e-81, 3.00956e-81, 3.17805e-81, 8.80785e-81};
  std::vector<double> out(probs.size());
  normalize_probabilities_safe(probs, kReachSmoothingEps, out.data());
  ASSERT_NEAR(vector_sum(out), 1.0, 1e-10);
}

TEST(UtilsTest, TestProbNormalizationFloat) {
  std::vector<double> probs{2.93185e-81, 3.00956e-81, 3.17805e-81, 8.80785e-81};
  std::vector<float> out(probs.size());
  normalize_probabilities_safe(probs, kReachSmoothingEps, out.data());
  ASSERT_NEAR(vector_sum(out), 1.0, 1e-10);
}