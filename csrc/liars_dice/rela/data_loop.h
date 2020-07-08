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

#include "net_interface.h"
#include "recursive_solving.h"
#include "rela/thread_loop.h"

namespace rela {

class CVNetBufferConnector : public IValueNet {
 public:
  CVNetBufferConnector(std::shared_ptr<ModelLocker> modelLocker,
                       std::shared_ptr<ValuePrioritizedReplay> replayBuffer)
      : modelLocker_(std::move(modelLocker)), replayBuffer_(replayBuffer) {}

  torch::Tensor compute_values(const torch::Tensor queries) {
    torch::NoGradGuard ng;
    const int kMaxSize = 1 << 12;
    const int size = queries.size(0);
    if (size > kMaxSize) {
      std::vector<int64_t> sizes;
      for (int start = 0; start < size; start += kMaxSize) {
        sizes.push_back(std::min(kMaxSize, size - start));
      }
      const auto sizesArray = c10::IntArrayRef(sizes.data(), sizes.size());
      auto chunks = torch::split_with_sizes(queries, sizesArray, 0);
      std::vector<torch::Tensor> results;
      for (auto input : chunks) {
        results.push_back(modelLocker_->forward(input));
      }
      return torch::cat(results, 0);
    } else {
      return modelLocker_->forward(queries);
    }
  }

  void add_training_example(const torch::Tensor queries,
                            const torch::Tensor values) {
    ValueTransition transition{queries, values};
    torch::Tensor priority = torch::ones(queries.size(0));
    replayBuffer_->add(transition, priority);
  }

  std::shared_ptr<ModelLocker> modelLocker_;
  std::shared_ptr<ValuePrioritizedReplay> replayBuffer_;
};

class DataThreadLoop : public ThreadLoop {
 public:
  DataThreadLoop(std::shared_ptr<CVNetBufferConnector> connector,
                 const liars_dice::RecursiveSolvingParams& cfg, int seed)
      : connector_(std::move(connector)), cfg_(cfg), seed_(seed) {}

  virtual void mainLoop() final {
    auto runner =
        std::make_unique<liars_dice::RlRunner>(cfg_, connector_, seed_);
    while (!terminated()) {
      if (paused()) {
        waitUntilResume();
      }
      runner->step();
    }
  }

 private:
  std::shared_ptr<IValueNet> connector_;
  const liars_dice::RecursiveSolvingParams cfg_;
  const int seed_;
};

}  // namespace rela
