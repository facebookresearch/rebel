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

#include <numeric>
#include <vector>

namespace liars_dice {

constexpr double kAlmostZero = 1e-200;

template <class T>
inline double normalize_probabilities(const std::vector<double>& unnormed_probs,
                                      T* probs) {
  const double sum =
      std::accumulate(unnormed_probs.begin(), unnormed_probs.end(), double{0});
  assert(sum >= kAlmostZero);
  for (size_t i = 0; i < unnormed_probs.size(); ++i) {
    probs[i] = unnormed_probs[i] / sum;
  }
  return sum;
}

inline double normalize_probabilities(const std::vector<double>& unnormed_probs,
                                      std::vector<double>* probs) {
  return normalize_probabilities(unnormed_probs, probs->data());
}

inline std::vector<double> normalize_probabilities(
    const std::vector<double>& unnormed_probs) {
  auto probs = unnormed_probs;
  normalize_probabilities(unnormed_probs, &probs);
  return probs;
}

template <class T>
inline double normalize_probabilities(const std::vector<double>& unnormed_probs,
                                      const std::vector<double>& last_probs,
                                      T* probs) {
  const double sum =
      std::accumulate(unnormed_probs.begin(), unnormed_probs.end(), double{0}) +
      std::accumulate(last_probs.begin(), last_probs.end(), double{0});
  assert(sum >= kAlmostZero);
  for (size_t i = 0; i < unnormed_probs.size(); ++i) {
    probs[i] = (unnormed_probs[i] + last_probs[i]) / sum;
  }
  return sum;
}

inline double normalize_probabilities(const std::vector<double>& unnormed_probs,
                                      const std::vector<double>& last_probs,
                                      std::vector<double>* probs) {
  return normalize_probabilities(unnormed_probs, last_probs, probs->data());
}

template <class T>
inline void normalize_probabilities_safe(
    const std::vector<double>& unnormed_probs, double eps, T* probs) {
  double sum = 0;
  for (size_t i = 0; i < unnormed_probs.size(); ++i) {
    sum += unnormed_probs[i] + eps;
  }
  for (size_t i = 0; i < unnormed_probs.size(); ++i) {
    probs[i] = (unnormed_probs[i] + eps) / sum;
  }
}

inline std::vector<double> normalize_probabilities_safe(
    const std::vector<double>& unnormed_probs, double eps) {
  std::vector<double> probs(unnormed_probs);
  normalize_probabilities_safe(unnormed_probs, eps, probs.data());
  return probs;
}

template <class T>
T vector_sum(const std::vector<T>& vector) {
  return std::accumulate(vector.begin(), vector.end(), T{0});
}

}  // namespace liars_dice