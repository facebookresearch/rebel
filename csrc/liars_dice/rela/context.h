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

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "rela/thread_loop.h"

namespace rela {

class Context {
 public:
  Context() : started_(false), numTerminatedThread_(0) {}

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  ~Context() {
    for (auto& v : loops_) {
      v->terminate();
    }
    for (auto& v : threads_) {
      v.join();
    }
  }

  int pushThreadLoop(std::shared_ptr<ThreadLoop> env) {
    assert(!started_);
    loops_.push_back(std::move(env));
    return (int)loops_.size();
  }

  void start() {
    for (int i = 0; i < (int)loops_.size(); ++i) {
      threads_.emplace_back([this, i]() {
        loops_[i]->mainLoop();
        ++numTerminatedThread_;
      });
    }
  }

  void pause() {
    for (auto& v : loops_) {
      v->pause();
    }
  }

  void resume() {
    for (auto& v : loops_) {
      v->resume();
    }
  }

  void terminate() {
    for (auto& v : loops_) {
      v->terminate();
    }
  }

  bool terminated() {
    // std::cout << ">>> " << numTerminatedThread_ << std::endl;
    return numTerminatedThread_ == (int)loops_.size();
  }

 private:
  bool started_;
  std::atomic<int> numTerminatedThread_;
  std::vector<std::shared_ptr<ThreadLoop>> loops_;
  std::vector<std::thread> threads_;
};
}  // namespace rela
