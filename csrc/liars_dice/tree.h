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

// Structures and functions for build a (partial) game tree.

#pragma once

#include <utility>
#include <vector>

#include "liars_dice.h"

namespace liars_dice {

struct UnrolledTreeNode;
using Tree = std::vector<UnrolledTreeNode>;

// The nodes are expected to be stored in a vector, with children_begin,
// children_end, and parent being indices in the vector.
struct UnrolledTreeNode {
  PartialPublicState state;
  int children_begin;
  int children_end;
  int parent;
  int depth;

  int num_children() const { return children_end - children_begin; }

  std::vector<int> get_children() const {
    std::vector<int> children(num_children());
    for (int i = 0; i < num_children(); ++i) {
      children[i] = children_begin + i;
    }
    return children;
  }
};

// Builds a BFS tree of this depth. For max_depth=0 the tree will contain only
// the root. For max_depth=1 - root and its children. And so on.
inline std::vector<UnrolledTreeNode> unroll_tree(const Game& game,
                                                 const PartialPublicState& root,
                                                 int max_depth) {
  assert(max_depth >= 0);  // Cannot build an empty tree.
  std::vector<UnrolledTreeNode> nodes;
  nodes.push_back(UnrolledTreeNode{root, 0, 0, -1, 0});
  for (int node_id = 0; node_id < static_cast<int>(nodes.size()) &&
                        nodes[node_id].depth < max_depth;
       ++node_id) {
    const auto [start, end] = game.get_bid_range(nodes[node_id].state);
    nodes.reserve(end - start + nodes.size());
    // No resizes beside this point.
    auto& parent = nodes[node_id];
    parent.children_begin = nodes.size();
    parent.children_end = parent.children_begin + end - start;
    for (int i = start; i < end; ++i) {
      auto state = game.act(parent.state, i);
      nodes.push_back(UnrolledTreeNode{state, 0, 0, node_id, parent.depth + 1});
    }
  }
  return nodes;
}

inline std::vector<UnrolledTreeNode> unroll_tree(const Game& game) {
  return unroll_tree(game, game.get_initial_state(), game.max_depth());
}

// Creates iterator over children nodes and corresponding actions.
// Usage:
//   for (auto[child_node_id, action] : ChildrenActionIt(node, game)) {
//      // do stuff
//   }
struct ChildrenActionIt {
  const Game& game;
  const UnrolledTreeNode& node;
  ChildrenActionIt(const UnrolledTreeNode& node, const Game& game)
      : game(game), node(node) {}
  struct State {
    int child;
    Action action;
    State(int child, Action action) : child(child), action(action) {}
    // Child node, action.
    std::pair<int, Action> operator*() const {
      return std::make_pair(child, action);
    }
    State& operator++() {
      ++child;
      ++action;
      return *this;
    }
    bool operator!=(const State& rhs) const { return child != rhs.child; }
  };
  State begin() const {
    return State(node.children_begin, game.get_bid_range(node.state).first);
  }
  State end() const {
    return State(node.children_end, game.get_bid_range(node.state).second);
  }
};

// Creates iterator over children nodes and corresponding actions.
// Usage:
//   for (auto child_node_id : ChildrenIt(node)) {
//      // do stuff
//   }
struct ChildrenIt {
  const UnrolledTreeNode& node;
  ChildrenIt(const UnrolledTreeNode& node) : node(node) {}
  struct State {
    int offset;
    State(int offset) : offset(offset) {}
    // Child node, action.
    int operator*() const { return offset; }
    State& operator++() {
      ++offset;
      return *this;
    }
    bool operator!=(const State& rhs) const { return offset != rhs.offset; }
  };
  State begin() const { return State(node.children_begin); }
  State end() const { return State(node.children_end); }
};
}  // namespace liars_dice
