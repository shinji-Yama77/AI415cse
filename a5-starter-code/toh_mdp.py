#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright by the University of Washington, 2021.

import dataclasses
import logging
import math
import random
from functools import cached_property
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)


@dataclasses.dataclass(eq=True, frozen=True)
class TohState:
    """Class representing the TOH MDP State.

    A TOH MDP state is the disks on each of the 3 pegs. The tuple for each peg
    represents the disks from bottom to top.
    """
    peg1: Tuple[int, ...]
    peg2: Tuple[int, ...]
    peg3: Tuple[int, ...]

    def __getitem__(self, peg: str) -> Tuple[int, ...]:
        return {
            "peg1": self.peg1,
            "peg2": self.peg2,
            "peg3": self.peg3,
        }[peg]

    def can_move(self, from_peg: str, to_peg: str) -> bool:
        from_peg_disks = self[from_peg]
        to_peg_disks = self[to_peg]
        if not from_peg_disks:
            return False
        if not to_peg_disks:
            return True
        if from_peg_disks[-1] < to_peg_disks[-1]:
            return True
        return False

    def move(self, from_peg: str, to_peg: str) -> "TohState":
        assert self.can_move(from_peg, to_peg)
        new_pegs = []
        for peg in ["peg1", "peg2", "peg3"]:
            if peg == from_peg:
                new_pegs.append(self[peg][:-1])
            elif peg == to_peg:
                new_pegs.append(self[peg] + (self[from_peg][-1],))
            else:
                new_pegs.append(self[peg])
        return TohState(*new_pegs)

    @classmethod
    def initial(cls, n_disks: int) -> "TohState":
        return TohState(tuple(range(n_disks, 0, -1)), (), ())

    @classmethod
    def goal(cls, n_disks: int) -> "TohState":
        return TohState((), (), tuple(range(n_disks, 0, -1)))

    @classmethod
    def goal2(cls, n_disks: int) -> "TohState":
        return TohState((n_disks,), (), tuple(range(n_disks - 1, 0, -1)))


@dataclasses.dataclass(eq=True, frozen=True)
class Operator:
    """Class representing the TOH MDP State.

    An Operator represents moving a peg from the `from_peg` to the `to_peg`.
    """
    from_peg: str
    to_peg: str

    @cached_property
    def name(self) -> str:
        return f"Move disk from {self.from_peg} to {self.to_peg}"

    def pre_condition(self, s: TohState) -> bool:
        return s.can_move(self.from_peg, self.to_peg)

    def state_transfer(self, s: TohState) -> TohState:
        return s.move(self.from_peg, self.to_peg)

    @cached_property
    def source_peg(self) -> int:
        return int(self.from_peg[-1])

    @cached_property
    def destination_peg(self) -> int:
        return int(self.to_peg[-1])

    @classmethod
    def all_operators(cls) -> List["Operator"]:
        peg_combinations = [
            (f"peg{a}", f"peg{b}") for a, b
            in [(1, 3), (1, 2), (3, 2), (3, 1), (2, 1), (2, 3)]]
        return [cls(p, q) for p, q in peg_combinations]

    @classmethod
    def leftward_operators(cls) -> List["Operator"]:
        peg_combinations = [
            (f"peg{a}", f"peg{b}") for a, b in [(1, 3), (3, 2), (2, 1)]]
        return [cls(p, q) for p, q in peg_combinations]

    @classmethod
    def rightward_operators(cls) -> List["Operator"]:
        peg_combinations = [
            (f"peg{a}", f"peg{b}") for a, b in [(1, 2), (2, 3), (3, 1)]]
        return [cls(p, q) for p, q in peg_combinations]


StateGraph = Dict[TohState, List[Tuple[Operator, TohState]]]
TohAction = str  # Action is simply the name of the operator.


def generate_all_states(
        initial: TohState, operators: List[Operator]
) -> Tuple[StateGraph, List[TohState]]:
    state_graph = {}
    open_states = [initial]
    closed_states = []

    while open_states:
        s = open_states.pop(0)
        closed_states.append(s)

        to_open = []
        neighbors = []

        for operator in operators:
            if operator.pre_condition(s):
                new_state = operator.state_transfer(s)
                neighbors.append((operator, new_state))
                if new_state not in closed_states:
                    to_open.append(new_state)
        state_graph[s] = neighbors
        open_states.extend(t for t in to_open if t not in open_states)

    return state_graph, closed_states


@dataclasses.dataclass(eq=True, frozen=True)
class TohMdpConfig:
    gamma: float
    living_reward: float
    noise: float
    n_disks: int
    n_goals: int

    def __post_init__(self) -> None:
        assert self.gamma <= 1.0, "gamma must be less than 1.0"
        assert self.n_goals in [1, 2], "Can only have 1 or 2 goal states."


@dataclasses.dataclass(frozen=True)
class TohMdp:
    config: TohMdpConfig
    operators: Tuple[Operator, ...]
    actions: Tuple[TohAction, ...]
    initial: TohState
    goal: TohState
    goal2: Optional[TohState]
    terminal: TohState
    state_graph: StateGraph
    nonterminal_states: Tuple[TohState, ...]
    rng: random.Random

    @classmethod
    def from_config(cls, config: TohMdpConfig) -> "TohMdp":
        operators = Operator.all_operators()
        actions = tuple(op.name for op in operators) + ('Exit',)

        initial = TohState.initial(config.n_disks)
        goal = TohState.goal(config.n_disks)
        goal2 = TohState.goal2(config.n_disks) if config.n_goals == 2 else None
        terminal = TohState((), (), ())

        state_graph, nonterminal_states = generate_all_states(initial, operators)

        random_seed = 0xdecafdad  # For generating random next states.

        return cls(config, tuple(operators), actions, initial, goal,
                   goal2, terminal, state_graph, tuple(nonterminal_states),
                   random.Random(random_seed))

    def is_goal(self, state: TohState) -> bool:
        return state == self.goal or (
                self.config.n_goals == 2 and state == self.goal2)

    def transition(self, state: TohState, action: TohAction, next_state: TohState) -> float:
        """The transition function for the MDP.

        The typical action is associated with one operator, and with the noise
        at 20% it has an 80% chance of having its effect produced by that operator.
        It has a 20% chance of "noise" which means all other possible next
        states (except Exit) operators share evenly in that probability.
        The Exit operator is the only allowable operator in the goal state(s).
        When a non-applicable operator is chosen by the agent, the effect will
        be 80% no-op (but living reward is taken), and a 20% chance that one of
        the applicable ops will be chosen (2 or 3)

        If no noise: Every applicable operator has its effect, and
        Every non applicable operator is a no-op.

        If noise, an action has 0.8 chance of it being applied and 0.2 chance
        that some other state is chosen at random from the set of remaining
        successors and the current state.

        Args:
            state: state before the action.
            action: action performed.
            next_state: the resulting state after performing the action.

        Returns:
            prob: float
                The probability of the transition, i.e., T(s, a, s') or
                P( s' | s, a ).
        """
        if action == "Exit" or self.is_goal(state):
            return 1.0 if next_state == self.terminal else 0.0

        if state == self.terminal:
            raise ValueError("Transition table for terminal state is undefined.")

        applicable_ops = [o for o in self.operators if o.pre_condition(state)]
        inapplicable_ops = [o for o in self.operators if not o.pre_condition(state)]

        possible_new_states = [o.state_transfer(state) for o in applicable_ops] + [state]
        for op, state_ in zip(applicable_ops, possible_new_states):
            if action == op.name and next_state == state_:
                return 1.0 - self.config.noise
        for op in inapplicable_ops:
            if action == op.name and next_state == state:
                return 1.0 - self.config.noise

        assert applicable_ops, f"{state} is non-goal states which should have applicable ops."
        noise_share = self.config.noise / (len(possible_new_states) - 1)
        if next_state in possible_new_states:
            return noise_share
        return 0.0

    def reward(self, state: TohState, action: TohAction, next_state: TohState) -> float:
        """The MDP Reward function.

        Rules: Exiting from the correct goal state yields a reward of +100.
        Exiting from an alternative goal state yields a reward of +10.
        The cost of living reward is defined in `config.living_reward`.

        Returns:
            reward: float
                The reward of the transition: R(s, a, s').
        """
        if state == self.goal:
            if action == "Exit" and next_state == self.terminal:
                return 100.0
            else:
                return 0.0
        elif self.config.n_goals == 2 and state == self.goal2:
            if action == "Exit" and next_state == self.terminal:
                return 10.0
            else:
                return 0.0
        return self.config.living_reward

    def make_solution_path(self, path_type: str) -> List[TohState]:
        assert path_type in ["golden", "silver"]
        logger.info("Looking for the %s path", path_type)
        parity = (self.config.n_disks + (path_type == "silver") + 1) % 2
        if parity:
            little_disk_ops = Operator.rightward_operators()
        else:
            little_disk_ops = Operator.leftward_operators()

        time_to_move_little_disk = True
        little_disk_op_idx = 0
        last_peg = 0
        s = self.nonterminal_states[0]
        path = [s]
        while not self.is_goal(s):
            if time_to_move_little_disk:
                op = little_disk_ops[little_disk_op_idx]
                little_disk_op_idx = (little_disk_op_idx + 1) % 3
            else:
                for i in range(6):
                    op = self.operators[i]
                    if op.source_peg == last_peg:
                        continue
                    if op.pre_condition(s):
                        break
            # noinspection PyUnboundLocalVariable
            if op.source_peg == last_peg:
                logger.info("No more moves for this path; it's probably the "
                            "path to the apex of the triangle.")
                return path
            new_state = op.state_transfer(s)
            last_peg = op.destination_peg
            path.append(new_state)
            time_to_move_little_disk = not time_to_move_little_disk
            s = new_state
        return path

    @cached_property
    def golden_path(self) -> List[TohState]:
        return self.make_solution_path("golden")

    @cached_property
    def silver_path(self) -> Optional[List[TohState]]:
        return (None if self.config.n_goals == 1
                else self.make_solution_path("silver"))

    @cached_property
    def all_states(self) -> List[TohState]:
        return list(self.nonterminal_states) + [self.terminal]

    def step(self, state: TohState, action: TohAction) -> Tuple[TohState, float]:
        """Simulate an MDP transition when at state and takes action.

        Args:
            state: the current state.
            action: the action taken.

        Returns:
            next_state: TohState
                The simulated next state.
            reward: float
                The simulated reward.
        """
        if self.is_goal(state):
            if action != "Exit":
                logger.warning(
                    "Action != Exit on a goal state %s will be interpreted as Exit", state)
            return self.terminal, self.reward(state, "Exit", self.terminal)
        if action == "Exit":
            return state, self.config.living_reward

        applicable_ops = [o for o in self.operators if o.pre_condition(state)]
        new_states = [o.state_transfer(state) for o in applicable_ops] + [state]
        new_state_probs = [self.transition(state, action, new_state)
                           for new_state in new_states]
        assert math.isclose(sum(new_state_probs), 1.0)
        new_state = self.rng.choices(new_states, new_state_probs)[0]
        return new_state, self.reward(state, action, new_state)


Policy = Dict[TohState, TohAction]
VTable = Dict[TohState, float]
QTable = Dict[Tuple[TohState, TohAction], float]
