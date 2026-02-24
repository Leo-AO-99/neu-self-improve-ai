"""
Minimal Monte Carlo Tree Search (MCTS) backbone for MCTS-RAG.
Based on https://github.com/yale-nlp/MCTS-RAG (run_src/MCTS_backbone.py).
"""
from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, TypeVar

NodeT = TypeVar("NodeT", bound="MCTS_Node")


class MCTS_Node(ABC):
    """Abstract MCTS node. Subclass to define find_children, is_terminal, calculate_reward."""

    _node_cnt = 0

    def __init__(self) -> None:
        super().__init__()
        MCTS_Node._node_cnt += 1
        self.id = MCTS_Node._node_cnt
        self.rollout_id: int | None = None

    def set_rollout_id(self, rollout_id: int) -> None:
        self.rollout_id = rollout_id

    @abstractmethod
    def find_children(self, rollout_id: int) -> List["MCTS_Node"]:
        """Return list of child nodes (possible successors)."""
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self) -> bool:
        """True if this node has no children (e.g. final answer)."""
        raise NotImplementedError

    @abstractmethod
    def calculate_reward(self) -> float:
        """Reward for this node (used when terminal)."""
        raise NotImplementedError

    def skip_backprop(self) -> bool:
        """If True, reward is not accumulated in backprop (default: False)."""
        return False


class MCTS_Searcher:
    """MCTS: rollout = select -> expand -> simulate -> backprop. Uses UCT for selection."""

    def __init__(
        self,
        exploration_weight: float = 2.0,
        weight_scheduler: str = "const",
        num_rollouts: int = 4,
        discount: float = 1.0,
        verbose: bool = False,
    ):
        self.Q: Dict[MCTS_Node, float] = defaultdict(lambda: 0.0)
        self.N: Dict[MCTS_Node, int] = defaultdict(lambda: 0)
        self.parent2children: Dict[MCTS_Node, List[MCTS_Node]] = {}
        self.explored_nodes: set = set()
        self.exploration_weight = exploration_weight
        self.weight_scheduler = weight_scheduler
        self.num_rollouts = num_rollouts
        self.discount = discount
        self.verbose = verbose

    def do_rollout(self, root_node: MCTS_Node, rollout_id: int) -> MCTS_Node | None:
        path_1 = self._select(root_node, rollout_id)
        leaf = path_1[-1]
        self._expand(leaf, rollout_id)
        path_2 = self._simulate(leaf, rollout_id)
        self._backpropagate(path_1 + path_2)
        try:
            return path_2[-1] if path_2 else path_1[-1]
        except IndexError:
            return path_1[-1] if path_1 else None

    def _select(self, node: MCTS_Node, rollout_id: int) -> List[MCTS_Node]:
        path = []
        while True:
            path.append(node)
            if node not in self.parent2children:
                return path
            unexplored = [n for n in self.parent2children[node] if n not in self.explored_nodes]
            if unexplored:
                n = random.choice(unexplored)
                path.append(n)
                return path
            node = self._uct_select(node, rollout_id)

    def _expand(self, node: MCTS_Node, rollout_id: int) -> None:
        if node in self.explored_nodes:
            return
        if node.is_terminal():
            self.explored_nodes.add(node)
            return
        self.parent2children[node] = node.find_children(rollout_id)

    def _simulate(self, node: MCTS_Node, rollout_id: int) -> List[MCTS_Node]:
        path = []
        cur = node
        while True:
            if cur.is_terminal():
                self.explored_nodes.add(node)
                return path
            if cur not in self.parent2children:
                self.parent2children[cur] = cur.find_children(rollout_id)
            cur = random.choice(self.parent2children[cur])
            path.append(cur)

    def _backpropagate(self, path: List[MCTS_Node]) -> None:
        if not path:
            return
        leaf = path[-1]
        reward = leaf.calculate_reward()
        for node in reversed(path):
            if not getattr(node, "skip_backprop", lambda: False)():
                self.Q[node] += reward
            self.N[node] += 1
            self.explored_nodes.add(node)

    def _get_weight(self, rollout_id: int) -> float:
        if self.weight_scheduler == "exp":
            return self.exploration_weight * (0.1 ** (rollout_id / max(1, self.num_rollouts)))
        if self.weight_scheduler == "lin":
            return self.exploration_weight * (1 - 0.9 * (rollout_id / max(1, self.num_rollouts)))
        return self.exploration_weight

    def _uct_select(self, node: MCTS_Node, rollout_id: int) -> MCTS_Node:
        children = self.parent2children[node]
        assert all(n in self.explored_nodes for n in children)
        return max(
            children,
            key=lambda n: self._compute_uct(parent_node=node, node=n, rollout_id=rollout_id),
        )

    def _compute_uct(self, parent_node: MCTS_Node, node: MCTS_Node, rollout_id: int) -> float:
        if self.N[node] == 0:
            return 999.0
        w = self._get_weight(rollout_id)
        return self.Q[node] / self.N[node] + w * math.sqrt(math.log(self.N[parent_node] + 1) / self.N[node])
