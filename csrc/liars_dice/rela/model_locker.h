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

#include <chrono>
#include <stack>
#include <thread>

#include <pybind11/pybind11.h>

#include "rela/types.h"

namespace rela {

template <class T>
class Stack {
 public:
  Stack() {}

  void push(T value) {
    {
      std::lock_guard<std::mutex> lk(m_);
      data_.push(value);
    }
    cv_.notify_one();
  }

  T pop() {
    std::unique_lock<std::mutex> lk(m_);
    cv_.wait(lk, [this] { return !data_.empty(); });
    auto value = data_.top();
    data_.pop();
    return value;
  }

 private:
  std::stack<T> data_;
  std::mutex m_;
  std::condition_variable cv_;
};

class ModelLocker {
 public:
  ModelLocker(std::vector<pybind11::object> pyModels, const std::string& device)
      : device(torch::Device(device)), pyModels_(pyModels) {
    for (size_t i = 0; i < pyModels_.size(); ++i) {
      models_.push_back(pyModels_[i].attr("_c").cast<TorchJitModel*>());
      availableModels_.push(i);
    }
  }

  ModelLocker(std::vector<TorchJitModel*> models, const std::string& device)
      : device(torch::Device(device)), models_(models) {
    for (size_t i = 0; i < models.size(); ++i) availableModels_.push(i);
  }

  void updateModel(pybind11::object pyModel) {
    for (size_t i = 0; i < pyModels_.size(); ++i) {
      availableModels_.pop();
    }
    for (auto& model : pyModels_) {
      model.attr("load_state_dict")(pyModel.attr("state_dict")());
    }
    for (size_t i = 0; i < pyModels_.size(); ++i) {
      availableModels_.push(i);
    }
  }

  int lock() { return availableModels_.pop(); }

  void unlock(int id) { availableModels_.push(id); }

  torch::Tensor forward(torch::Tensor query, int model_id = -1) {
    const bool lock = model_id == -1;
    const int id = lock ? availableModels_.pop() : model_id;
    std::vector<torch::jit::IValue> inputs = {query.to(device)};
    auto results = models_[id]->forward(inputs);
    // Detach is needed to free the memory allocated to gradients. Either this
    // or torch::NoGradGuard.
    auto results_cpu = torch::detach(results.toTensor().to(torch::kCPU));
    if (lock) availableModels_.push(id);
    return results_cpu;
  }

  const torch::Device device;

 private:
  std::vector<pybind11::object> pyModels_;
  std::vector<TorchJitModel*> models_;
  Stack<int> availableModels_;
};

}  // namespace rela
