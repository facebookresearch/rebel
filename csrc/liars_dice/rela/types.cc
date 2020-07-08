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

#include "rela/types.h"

using namespace rela;

ValueTransition ValueTransition::makeBatch(
    const std::vector<ValueTransition>& transitions,
    const std::string& device) {
  std::vector<torch::Tensor> queryVec;
  std::vector<torch::Tensor> valuesVec;

  for (size_t i = 0; i < transitions.size(); i++) {
    queryVec.push_back(transitions[i].query);
    valuesVec.push_back(transitions[i].values);
  }

  ValueTransition batch;
  batch.query = torch::stack(queryVec, 0);
  batch.values = torch::stack(valuesVec, 0);

  if (device != "cpu") {
    auto d = torch::Device(device);
    batch.query = batch.query.to(d);
    batch.values = batch.values.to(d);
  }

  return batch;
}

std::vector<torch::Tensor> ValueTransition::toVector() {
  return std::vector<torch::Tensor>{query, values};
}

ValueTransition ValueTransition::fromVector(
    const std::vector<torch::Tensor>& tensors) {
  ValueTransition result;
  assert(tensors.size() == 2);
  result.query = tensors[0];
  result.values = tensors[1];
  return std::move(result);
}

ValueTransition ValueTransition::index(int i) const {
  ValueTransition element;

  element.query = query[i];
  element.values = values[i];

  return element;
}

ValueTransition ValueTransition::padLike() const {
  ValueTransition pad;

  pad.query = torch::zeros_like(query);
  pad.values = torch::zeros_like(values);

  return pad;
}

TorchJitInput ValueTransition::toJitInput(const torch::Device& device) const {
  TorchJitInput input;
  input.push_back(query.to(device));
  input.push_back(values.to(device));
  return input;
}

float readInt(FILE* file) {
  int tmp;
  fread(&tmp, sizeof(int), 1, file);
  return static_cast<float>(tmp);
}

void ValueTransition::write(FILE* file) const {
  const int query_size = query.numel();
  const int value_size = values.numel();
  fwrite(&query_size, sizeof(int), 1, file);
  fwrite(&value_size, sizeof(int), 1, file);
  fwrite(query.data_ptr<float>(), sizeof(float), query.numel(), file);
  fwrite(values.data_ptr<float>(), sizeof(float), values.numel(), file);
}

ValueTransition ValueTransition::load(FILE* file, bool* success) {
  ValueTransition result;
  *success = true;
  int query_size, value_size;
  if (!fread(&query_size, sizeof(int), 1, file)) {
    *success = false;
    return result;
  }
  fread(&value_size, sizeof(int), 1, file);
  result.query = torch::zeros({1, query_size});
  result.values = torch::zeros({1, value_size});
  fread(result.query.data_ptr<float>(), sizeof(float), query_size, file);
  assert(
      fread(result.values.data_ptr<float>(), sizeof(float), value_size, file));
  return result;
}
