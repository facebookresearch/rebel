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

#include <torch/torch.h>

// Interface for the value function network.
class IValueNet {
 public:
  //   virtual torch::Device get_device() const = 0;

  virtual ~IValueNet() = default;

  // Passes a query tensor [batch, query_dim] to the net and returns expected
  // values [batch, belief_size].
  virtual torch::Tensor compute_values(const torch::Tensor queries) = 0;

  // Callback to pass the true value for the query to the trainer.
  virtual void add_training_example(const torch::Tensor queries,
                                    const torch::Tensor values) = 0;
};
