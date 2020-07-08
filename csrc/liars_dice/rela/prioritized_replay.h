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

#include <stdio.h>
#include <future>
#include <random>
#include <vector>

#include <torch/extension.h>

#include "rela/types.h"

namespace rela {

using ExtractedData = std::vector<torch::Tensor>;

template <class DataType>
class ConcurrentQueue {
 public:
  ConcurrentQueue(int capacity)
      : capacity(capacity),
        head_(0),
        tail_(0),
        size_(0),
        allow_write_(true),
        safeTail_(0),
        safeSize_(0),
        sum_(0),
        evicted_(capacity, false),
        elements_(capacity),
        weights_(capacity, 0) {}

  int safeSize(float* sum) const {
    std::unique_lock<std::mutex> lk(m_);
    if (sum != nullptr) {
      *sum = sum_;
    }
    return safeSize_;
  }

  int size() const {
    std::unique_lock<std::mutex> lk(m_);
    return size_;
  }

  void blockAppend(const std::vector<DataType>& block,
                   const torch::Tensor& weights) {
    int blockSize = block.size();

    std::unique_lock<std::mutex> lk(m_);
    cvSize_.wait(lk,
                 [=] { return size_ + blockSize <= capacity && allow_write_; });

    int start = tail_;
    int end = (tail_ + blockSize) % capacity;

    tail_ = end;
    size_ += blockSize;
    checkSize(head_, tail_, size_);

    lk.unlock();

    float sum = 0;
    auto weightAcc = weights.accessor<float, 1>();
    assert(weightAcc.size(0) == blockSize);
    for (int i = 0; i < blockSize; ++i) {
      int j = (start + i) % capacity;
      elements_[j] = block[i];
      weights_[j] = weightAcc[i];
      sum += weightAcc[i];
    }

    lk.lock();

    cvTail_.wait(lk, [=] { return safeTail_ == start; });
    safeTail_ = end;
    safeSize_ += blockSize;
    sum_ += sum;
    checkSize(head_, safeTail_, safeSize_);

    lk.unlock();
    cvTail_.notify_all();
  }

  // ------------------------------------------------------------- //
  // blockPop, update are thread-safe against blockAppend
  // but they are NOT thread-safe against each other

  void blockPop(int blockSize) {
    double diff = 0;
    int head = head_;
    for (int i = 0; i < blockSize; ++i) {
      diff -= weights_[head];
      evicted_[head] = true;
      head = (head + 1) % capacity;
    }

    {
      std::lock_guard<std::mutex> lk(m_);
      sum_ += diff;
      head_ = head;
      safeSize_ -= blockSize;
      size_ -= blockSize;
      assert(safeSize_ >= 0);
      checkSize(head_, safeTail_, safeSize_);
    }
    cvSize_.notify_all();
  }

  void save(const std::string& fpath) {
    std::lock_guard<std::mutex> lk(m_);
    FILE* stream = fopen(fpath.c_str(), "wb");
    for (int i = 0; i < size_; ++i) {
      elements_[i].write(stream);
    }
    fclose(stream);
  }

  ExtractedData extract() {
    std::cerr << "Starting extract" << std::endl;
    const int size = safeSize_;

    // Create data dump.
    std::vector<DataType> data;
    std::vector<float> weights;
    data.reserve(size);
    for (int i = 0; i < size; ++i) {
      const auto index = (i + head_) % capacity;
      data.push_back(elements_[index]);
      weights.push_back(weights_[index]);
    }
    torch::Tensor weights_tensor =
        torch::from_blob(weights.data(), {(long long)weights.size()}).clone();
    auto batched = DataType::makeBatch(data, "cpu").toVector();

    blockPop(size);
    batched.push_back(weights_tensor);
    return batched;
  }

  void update(const std::vector<int>& ids, const torch::Tensor& weights) {
    double diff = 0;
    auto weightAcc = weights.accessor<float, 1>();
    for (int i = 0; i < (int)ids.size(); ++i) {
      auto id = ids[i];
      if (evicted_[id]) {
        continue;
      }
      diff += (weightAcc[i] - weights_[id]);
      weights_[id] = weightAcc[i];
    }

    std::lock_guard<std::mutex> lk_(m_);
    sum_ += diff;
  }

  // ------------------------------------------------------------- //
  // accessing elements is never locked, operate safely!

  DataType getElementAndMark(int idx) {
    int id = (head_ + idx) % capacity;
    evicted_[id] = false;
    return elements_[id];
  }

  float getWeight(int idx, int* id) {
    assert(id != nullptr);
    *id = (head_ + idx) % capacity;
    return weights_[*id];
  }

  const int capacity;

 private:
  void checkSize(int head, int tail, int size) {
    if (size == 0) {
      assert(tail == head);
    } else if (tail > head) {
      if (tail - head != size) {
        std::cout << "tail-head: " << tail - head << " vs size: " << size
                  << std::endl;
      }
      assert(tail - head == size);
    } else {
      if (tail + capacity - head != size) {
        std::cout << "tail-head: " << tail + capacity - head
                  << " vs size: " << size << std::endl;
      }
      assert(tail + capacity - head == size);
    }
  }

  mutable std::mutex m_;
  std::condition_variable cvSize_;
  std::condition_variable cvTail_;

  int head_;
  int tail_;
  int size_;
  std::atomic_bool allow_write_;

  int safeTail_;
  int safeSize_;
  double sum_;
  std::vector<bool> evicted_;

  std::vector<DataType> elements_;
  std::vector<float> weights_;
};

template <class DataType>
class PrioritizedReplay {
 public:
  PrioritizedReplay(int capacity, int seed, float alpha, float beta,
                    int prefetch, bool use_priority,
                    bool compressed_values = false)
      : alpha_(alpha)  // priority exponent
        ,
        beta_(beta)  // importance sampling exponent
        ,
        prefetch_(prefetch),
        capacity_(capacity),
        use_priority_(use_priority),
        compressed_values_(compressed_values),
        storage_(int(1.25 * capacity)),
        numAdd_(0) {
    rng_.seed(seed);
  }
  PrioritizedReplay(int capacity, int seed, float alpha, float beta,
                    int prefetch)
      : PrioritizedReplay(capacity, seed, alpha, beta, prefetch,
                          /*use_priority=*/true) {}

  void add(const std::vector<DataType>& sample, const torch::Tensor& priority) {
    assert(priority.dim() == 1);
    auto weights = use_priority_ ? torch::pow(priority, alpha_) : priority;
    storage_.blockAppend(sample, weights);
    numAdd_ += priority.size(0);
  }

  void add(const DataType& sample, const torch::Tensor& priority) {
    std::vector<DataType> vec;
    int n = priority.size(0);
    for (int i = 0; i < n; ++i) {
      vec.push_back(sample.index(i));
    }
    add(vec, priority);
  }

  std::tuple<DataType, torch::Tensor> sample(int batchsize,
                                             const std::string& device) {
    if (!sampledIds_.empty()) {
      if (use_priority_) {
        std::cout << "Error: previous samples' priority has not been updated."
                  << std::endl;
        assert(false);
      }
    }

    DataType batch;
    torch::Tensor priority;
    if (prefetch_ == 0) {
      std::tie(batch, priority, sampledIds_) = sample_(batchsize, device);
      return std::make_tuple(batch, priority);
    }

    if (futures_.empty()) {
      std::tie(batch, priority, sampledIds_) = sample_(batchsize, device);
    } else {
      // assert(futures_.size() == 1);
      std::tie(batch, priority, sampledIds_) = futures_.front().get();
      futures_.pop();
    }

    while ((int)futures_.size() < prefetch_) {
      auto f =
          std::async(std::launch::async, &PrioritizedReplay<DataType>::sample_,
                     this, batchsize, device);
      futures_.push(std::move(f));
    }

    return std::make_tuple(batch, priority);
  }

  void updatePriority(const torch::Tensor& priority) {
    if (priority.size(0) == 0) {
      sampledIds_.clear();
      return;
    }

    assert(priority.dim() == 1);
    assert((int)sampledIds_.size() == priority.size(0));

    auto weights = torch::pow(priority, alpha_);
    {
      std::lock_guard<std::mutex> lk(mSampler_);
      storage_.update(sampledIds_, weights);
    }
    sampledIds_.clear();
  }

  int size() const { return storage_.safeSize(nullptr); }

  int numAdd() const { return numAdd_; }

  void load(const std::string& fpath, float priority, int max_size,
            int stride) {
    FILE* stream = fopen(fpath.c_str(), "rb");
    torch::Tensor priority_tensor = torch::ones(1) * priority;
    for (int added = 0, i = 0;; ++i) {
      if (max_size > 0 && added == max_size) break;
      bool success;
      DataType data = DataType::load(stream, &success);
      if (!success) break;
      if (i % stride != 0) continue;
      add(data, priority_tensor);
      ++added;
    }
    fclose(stream);
  }

  void save(const std::string& fpath) { storage_.save(fpath); }

  // Get context of the buffer as a vector of tensors.
  ExtractedData extract() {
    // Taking just in case. If this methood is used callers are not expected to
    // sample.
    std::lock_guard<std::mutex> lk(mSampler_);
    auto data = storage_.extract();
    data.back() = torch::pow(data.back(), 1 / alpha_);
    return data;
  }

  // Push content extracted from another buffer.
  void push(ExtractedData data) {
    const torch::Tensor weights = data.back();
    data.pop_back();
    const DataType elements = DataType::fromVector(data);
    add(elements, weights);
  }

  // Pop from the buffer until new_size is left.
  void popUntil(int new_size) {
    auto size = storage_.size();
    if (size > new_size) {
      storage_.blockPop(size - new_size);
    }
  }

 private:
  using SampleWeightIds = std::tuple<DataType, torch::Tensor, std::vector<int>>;

  SampleWeightIds sample_(int batchsize, const std::string& device) {
    if (use_priority_) {
      return sample_with_priorities_(batchsize, device);
    } else {
      return sample_no_priorities_(batchsize, device);
    }
  }

  SampleWeightIds sample_with_priorities_(int batchsize,
                                          const std::string& device) {
    std::unique_lock<std::mutex> lk(mSampler_);

    float sum;
    int size = storage_.safeSize(&sum);
    // std::cout << "size: "<< size << ", sum: " << sum << std::endl;
    // storage_ [0, size) remains static in the subsequent section

    float segment = sum / batchsize;
    std::uniform_real_distribution<float> dist(0.0, segment);

    std::vector<DataType> samples;
    auto weights = torch::zeros({batchsize}, torch::kFloat32);
    auto weightAcc = weights.accessor<float, 1>();
    std::vector<int> ids(batchsize);

    double accSum = 0;
    int nextIdx = 0;
    float w = 0;
    int id = 0;
    for (int i = 0; i < batchsize; i++) {
      float rand = dist(rng_) + i * segment;
      rand = std::min(sum - (float)0.1, rand);
      // std::cout << "looking for " << i << "th/" << batchsize << " sample" <<
      // std::endl;
      // std::cout << "\ttarget: " << rand << std::endl;

      while (nextIdx <= size) {
        if ((accSum > 0 && accSum >= rand) || nextIdx == size) {
          assert(nextIdx >= 1);
          // std::cout << "\tfound: " << nextIdx - 1 << ", " << id << ", " <<
          // accSum << std::endl;
          DataType element = storage_.getElementAndMark(nextIdx - 1);
          samples.push_back(element);
          weightAcc[i] = w;
          ids[i] = id;
          break;
        }

        if (nextIdx == size) {
          // This should never happened due to the hackky if above.
          std::cout << "nextIdx: " << nextIdx << "/" << size << std::endl;
          std::cout << std::setprecision(10) << "accSum: " << accSum
                    << ", sum: " << sum << ", rand: " << rand << std::endl;
          assert(false);
        }

        w = storage_.getWeight(nextIdx, &id);
        accSum += w;
        ++nextIdx;
      }
    }
    assert((int)samples.size() == batchsize);

    // pop storage if full
    size = storage_.size();
    if (size > capacity_) {
      storage_.blockPop(size - capacity_);
    }

    // safe to unlock, because <samples> contains copys
    lk.unlock();

    weights = weights / sum;
    weights = torch::pow(size * weights, -beta_);
    weights /= weights.max();
    if (device != "cpu") {
      weights = weights.to(torch::Device(device));
    }
    auto batch = DataType::makeBatch(samples, device);
    if (compressed_values_) {
      batch.values = rela::dequantize(batch.values);
    }
    return std::make_tuple(batch, weights, ids);
  }

  SampleWeightIds sample_no_priorities_(int batchsize,
                                        const std::string& device) {
    std::unique_lock<std::mutex> lk(mSampler_);

    int size = storage_.safeSize(nullptr);
    // std::cout << "size: "<< size << ", sum: " << sum << std::endl;
    // storage_ [0, size) remains static in the subsequent section

    std::uniform_int_distribution<> dist(0, size - 1);

    std::vector<DataType> samples;
    auto weights = torch::zeros({batchsize}, torch::kFloat32);
    auto weightAcc = weights.accessor<float, 1>();
    std::vector<int> ids(batchsize);
    for (int i = 0; i < batchsize; i++) {
      const int index = dist(rng_);
      weightAcc[i] = storage_.getWeight(index, &ids[i]);
      DataType element = storage_.getElementAndMark(index);
      samples.push_back(element);
    }
    assert((int)samples.size() == batchsize);

    // pop storage if full
    size = storage_.size();
    if (size > capacity_) {
      storage_.blockPop(size - capacity_);
    }

    // safe to unlock, because <samples> contains copys
    lk.unlock();
    auto batch = DataType::makeBatch(samples, device);
    if (compressed_values_) {
      batch.values = rela::dequantize(batch.values);
    }
    return std::make_tuple(batch, weights, ids);
  }

  const float alpha_;
  const float beta_;
  const int prefetch_;
  const int capacity_;
  const bool use_priority_;
  const bool compressed_values_;

  ConcurrentQueue<DataType> storage_;
  std::atomic<int> numAdd_;

  // make sure that sample & update does not overlap
  std::mutex mSampler_;
  std::vector<int> sampledIds_;
  std::queue<std::future<SampleWeightIds>> futures_;

  std::mt19937 rng_;
};

using ValuePrioritizedReplay = PrioritizedReplay<ValueTransition>;
}  // namespace rela
