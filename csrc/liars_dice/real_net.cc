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

#include "real_net.h"

#include <iostream>
#include <memory>

#include <torch/script.h>
#include <torch/torch.h>

#include "liars_dice.h"
#include "net_interface.h"
#include "subgame_solving.h"

namespace liars_dice {
namespace {

class ZeroOutputNet : public IValueNet {
 public:
  ZeroOutputNet(int output_size, bool verbose)
      : output_size_(output_size), verbose_(verbose) {}

  torch::Tensor compute_values(const torch::Tensor query) override {
    const int num_queries = query.size(0);
    if (verbose_) {
      std::cerr << "Called ZeroOutputNet::handle_nn_query() with num_queries="
                << num_queries << std::endl;
    }
    return torch::zeros({num_queries, output_size_});
  }

  void add_training_example(const torch::Tensor query,
                            const torch::Tensor /*values*/) override {
    if (verbose_) {
      std::cerr << "Called ZeroOutputNet::cvfnet_update() with num_entries="
                << query.size(0) << std::endl;
    }
  }

 private:
  int output_size_;
  bool verbose_;
};

class TorchScriptNet : public IValueNet {
 public:
  TorchScriptNet(const std::string& path, const std::string& device)
      : device_(device) {
    try {
      // Deserialize the ScriptModule from a file using torch::jit::load().
      module_ = torch::jit::load(path);
    } catch (const c10::Error& e) {
      std::cerr << "error loading the model: " << path << std::endl;
      std::cerr << e.what();
      return;
    }
    std::cerr << "Loaded: " << path << std::endl;
    module_.to(device_);
  }

  torch::Tensor compute_values(const torch::Tensor query) override {
    std::vector<torch::jit::IValue> inputs = {query.to(device_)};
    auto results = module_.forward(inputs);
    return results.toTensor().to(torch::kCPU);
  }

  void add_training_example(const torch::Tensor /*query*/,
                            const torch::Tensor /*values*/) override {
    throw std::runtime_error("Cannot update TorchScript model, only query");
  }

 private:
  torch::jit::script::Module module_;
  const torch::Device device_;
};

class OracleNetSolver : public IValueNet {
 public:
  OracleNetSolver(const Game& game, const SubgameSolvingParams& params)
      : game(game), params(params) {}

  torch::Tensor compute_values(const torch::Tensor queries) override {
    const int num_queries = queries.size(0);
    std::vector<torch::Tensor> values;
    for (int query_id = 0; query_id < num_queries; ++query_id) {
      auto row = queries[query_id];
      auto row_values = compute_values(row.data_ptr<float>());
      values.push_back(torch::tensor(row_values).to(torch::kFloat32));
    }
    return torch::stack(values, 0);
  }

  // Callback to pass the true value for the query to the trainer.
  void add_training_example(const torch::Tensor /*queries*/,
                            const torch::Tensor /*values*/) final {
    throw "not supported";
  }

 private:
  std::vector<double> compute_values(const float* query) {
    auto [traverser, state, beliefs1, beliefs2] =
        deserialize_query(game, query);
    Pair<std::vector<double>> beliefs = {beliefs1, beliefs2};
    auto solver = build_solver(game, state, beliefs, params, /*net=*/nullptr);
    solver->multistep();
    return solver->get_hand_values(traverser);
  }

  const Game game;
  const SubgameSolvingParams params;
};
}  // namespace

std::shared_ptr<IValueNet> create_zero_net(int output_size, bool verbose) {
  return std::make_shared<ZeroOutputNet>(output_size, verbose);
}

std::shared_ptr<IValueNet> create_torchscript_net(const std::string& path) {
  return std::make_shared<TorchScriptNet>(path, "cuda");
}
std::shared_ptr<IValueNet> create_torchscript_net(const std::string& path,
                                                  const std::string& device) {
  return std::make_shared<TorchScriptNet>(path, device);
}

std::shared_ptr<IValueNet> create_oracle_value_predictor(
    const Game& game, const SubgameSolvingParams& params) {
  return std::make_shared<OracleNetSolver>(game, params);
}

}  // namespace liars_dice