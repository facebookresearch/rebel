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

#include <memory>

#include "liars_dice.h"
#include "net_interface.h"
#include "subgame_solving.h"

namespace liars_dice {

float eval_net(const Game& game, const TreeStrategy& net_strategy,
               const TreeStrategy& full_strategy, int mdp_depth, int fp_iters,
               std::shared_ptr<IValueNet> net, bool traverse_by_net,
               bool verbose);

}  // namespace liars_dice