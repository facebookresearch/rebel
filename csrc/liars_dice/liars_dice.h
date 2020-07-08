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

#pragma once

#include <assert.h>

#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace liars_dice {

using Action = int;

// All actions, but the liar call, could be represented as (quantity, face)
// pair.
struct UnpackedAction {
  int quantity, face;
};

// Public state of the game without tracking the history of the game.
struct PartialPublicState {
  // Previous call.
  Action last_bid;
  // Player to make move next.
  int player_id;

  bool operator==(const PartialPublicState& state) const {
    return last_bid == state.last_bid && player_id == state.player_id;
  }
};

class Game {
 public:
  const int num_dice;
  const int num_faces;

  Game(int num_dice, int num_faces)
      : num_dice(num_dice),
        num_faces(num_faces),
        total_num_dice_(num_dice * 2),
        num_actions_(1 + total_num_dice_ * num_faces),
        num_hands_(int_pow(num_faces, num_dice)),
        liar_call_(num_actions_ - 1),
        wild_face_(num_faces - 1) {}

  // Number of dice for all the players.
  int total_num_dice() const { return total_num_dice_; }
  // Maximum number of distinct actions in every node.
  Action num_actions() const { return num_actions_; }
  // Number of distrinct game states at the beginning of the game. In other
  // words, number of different realization of the chance nodes.
  int num_hands() const { return num_hands_; }
  // Action id for Liar call.
  Action liar_call() const { return liar_call_; }
  // Id for the "wild" face.
  int wild_face() const { return wild_face_; }
  // Upper bound for how deep game tree could be.
  int max_depth() const { return 1 + num_actions_; }

  UnpackedAction unpack_action(Action action) const {
    assert(action != liar_call() && action != kInitialAction);
    UnpackedAction unpacked_action;
    unpacked_action.quantity = 1 + action / num_faces;
    unpacked_action.face = action % num_faces;
    return unpacked_action;
  }

  // Return number of dice in the hand that match the face.
  int num_matches(int hand, int face) const {
    int matches = 0;
    for (int i = 0; i < num_dice; ++i) {
      int dice_face = hand % num_faces;
      matches += (dice_face == face || dice_face == wild_face());
      hand /= num_faces;
    }
    return matches;
  }
  // Simple wrapper to compute num matches for all faces. Slow. For debugging
  // only.
  std::vector<int> num_matches(int hand) const {
    std::vector<int> matches;
    for (int i = 0; i < num_faces; ++i) {
      matches.push_back(num_matches(hand, i));
    }
    return matches;
  }

  PartialPublicState get_initial_state() const {
    PartialPublicState state;
    state.last_bid = kInitialAction;
    state.player_id = 0;
    return state;
  }

  // Get range of possible actions in the state as [min_action, max_action).
  std::pair<Action, Action> get_bid_range(
      const PartialPublicState& state) const {
    return state.last_bid == kInitialAction
               ? std::pair<Action, Action>(0, num_actions() - 1)
               : std::pair<Action, Action>(state.last_bid + 1, num_actions());
  }

  bool is_terminal(const PartialPublicState& state) const {
    return state.last_bid == liar_call();
  }

  PartialPublicState act(const PartialPublicState& state, Action action) const {
    const auto range = get_bid_range(state);
    assert(action >= range.first);
    assert(action < range.second);
    PartialPublicState new_state;
    new_state.last_bid = action;
    new_state.player_id = 1 - state.player_id;
    return new_state;
  }

  std::string action_to_string(Action action) const;
  std::string state_to_string(const PartialPublicState& state) const;
  std::string action_to_string_short(Action action) const;
  std::string state_to_string_short(const PartialPublicState& state) const;

 private:
  static int int_pow(int base, int power) {
    if (power == 0) return 1;
    const int half_power = int_pow(base, power / 2);
    const int reminder = (power % 2 == 0) ? 1 : base;
    const double double_half_power = half_power;
    const double double_result =
        double_half_power * double_half_power * reminder;
    assert(double_result + 1 <
           static_cast<double>(std::numeric_limits<int>::max()));
    return half_power * half_power * reminder;
  }

  static constexpr int kInitialAction = -1;
  const int total_num_dice_;
  const Action num_actions_;
  const int num_hands_;
  const Action liar_call_;
  const int wild_face_;
};

}  // namespace liars_dice