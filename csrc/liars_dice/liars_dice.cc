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

#include "liars_dice.h"
#include "real_net.h"

#include <sstream>

namespace liars_dice {
std::string Game::action_to_string(Action action) const {
  if (action == liar_call()) {
    return "liar";
  }
  std::ostringstream ss;
  const auto unpacked = unpack_action(action);
  ss << "bid(quantity=" << unpacked.quantity << ",face=" << unpacked.face
     << ")";
  return ss.str();
}

std::string Game::state_to_string(const PartialPublicState& state) const {
  std::ostringstream ss;
  const std::string last_bid = state.last_bid == kInitialAction
                                   ? "start"
                                   : action_to_string(state.last_bid);
  ss << "(pid=" << state.player_id << ",last=" << last_bid << ")";
  return ss.str();
}

std::string Game::action_to_string_short(Action action) const {
  if (action == liar_call()) {
    return "liar";
  }
  std::ostringstream ss;
  const auto unpacked = unpack_action(action);
  ss << unpacked.quantity << "x" << unpacked.face << "f";
  return ss.str();
}

std::string Game::state_to_string_short(const PartialPublicState& state) const {
  std::ostringstream ss;
  const std::string last_bid = state.last_bid == kInitialAction
                                   ? "beg"
                                   : action_to_string_short(state.last_bid);
  ss << "p" << state.player_id << "," << last_bid;
  return ss.str();
}
}  // namespace liars_dice