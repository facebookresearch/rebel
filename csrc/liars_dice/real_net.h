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

// Creates a net that outputs zeros on query and nothing on update.
std::shared_ptr<IValueNet> create_zero_net(int output_size,
                                           bool verbose = true);

// Creat eval-only connector from the net in the path.
std::shared_ptr<IValueNet> create_torchscript_net(const std::string& path);
std::shared_ptr<IValueNet> create_torchscript_net(const std::string& path,
                                                  const std::string& device);

// Create virtual value net that run a solver for each query.
std::shared_ptr<IValueNet> create_oracle_value_predictor(
    const Game& game, const SubgameSolvingParams& params);

}  // namespace liars_dice