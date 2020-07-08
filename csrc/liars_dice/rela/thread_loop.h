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
#include <chrono>

#include "rela/model_locker.h"
#include "rela/prioritized_replay.h"
#include "rela/types.h"

namespace rela {

class ThreadLoop {
 public:
  ThreadLoop() = default;

  ThreadLoop(const ThreadLoop&) = delete;
  ThreadLoop& operator=(const ThreadLoop&) = delete;

  virtual ~ThreadLoop() {}

  virtual void terminate() { terminated_ = true; }

  virtual void pause() {
    std::lock_guard<std::mutex> lk(mPaused_);
    paused_ = true;
  }

  virtual void resume() {
    {
      std::lock_guard<std::mutex> lk(mPaused_);
      paused_ = false;
    }
    cvPaused_.notify_one();
  }

  virtual void waitUntilResume() {
    std::unique_lock<std::mutex> lk(mPaused_);
    cvPaused_.wait(lk, [this] { return !paused_; });
  }

  virtual bool terminated() { return terminated_; }

  virtual bool paused() { return paused_; }

  virtual void mainLoop() = 0;

 private:
  std::atomic_bool terminated_{false};

  std::mutex mPaused_;
  bool paused_ = false;
  std::condition_variable cvPaused_;
};

}  // namespace rela
