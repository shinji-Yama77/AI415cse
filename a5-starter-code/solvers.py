#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright by the University of Washington, 2021.

import abc
import logging
import random
from typing import Optional, List

import solver_utils
import toh_mdp as tm

logger = logging.getLogger(__name__)


class TabularSolver(metaclass=abc.ABCMeta):
    """Base class for tabular based reinforcement learning MDP solvers.

    For details on tabular solution methods, see Sutton & Barto's 2018
    book on reinforcement learning, which devotes its first part to tabular
    methods.
    """
    def __init__(self, mdp: tm.TohMdp):
        self.mdp = mdp
        self._v_table: tm.VTable = {}
        self._q_table: tm.QTable = {}
        self._reset_v_table(0.0)
        self._reset_q_table(0.0)

    def _reset_v_table(self, value: float) -> None:
        for s in self.mdp.all_states:
            self._v_table[s] = value

    def _reset_q_table(self, value: float) -> None:
        for s in self.mdp.nonterminal_states:
            for a in self.mdp.actions:
                self._q_table[(s, a)] = value

    @property
    @abc.abstractmethod
    def policy(self) -> tm.Policy:
        """Maps nonterminal states to actions."""
        raise NotImplementedError

    def reset(self) -> None:
        self._reset_v_table(0.0)
        self._reset_q_table(0.0)

    @property
    def v_table(self) -> tm.VTable:
        return self._v_table

    @property
    def q_table(self) -> tm.QTable:
        return self._q_table


class ValueIterationSolver(TabularSolver):
    """Value Iteration MDP Solver."""
    TOLERANCE = 1e-8

    def __init__(self, mdp: tm.TohMdp):
        super().__init__(mdp)
        self.n_iteration = 0
        self.converged = False
        logger.info(f"{self.__class__.__name__} initialized with {mdp.config}")

    @property
    def policy(self) -> tm.Policy:
        return solver_utils.extract_policy(self.mdp, self.q_table)

    def step(self) -> float:
        if self.converged:
            logger.info("VI Solver has already converged!")
            return 0.0

        self.n_iteration += 1

        new_v_table, q_table, max_delta = solver_utils.value_iteration(
            self.mdp, self.v_table)

        self._v_table = new_v_table
        self._q_table = q_table
        logger.info("After %s iterations, max_delta = %s", self.n_iteration, max_delta)
        if max_delta < self.TOLERANCE:
            self.converged = True
            logger.info("VI Solver converged after %s iterations.", self.n_iteration)
        return max_delta

    def reset(self) -> None:
        super().reset()
        self.n_iteration = 0
        self.converged = False


class QLearningSolver(TabularSolver):
    """Q Learning MDP solver.

    Note that self.alpha and self.epsilon will be set toNone when the solver is
    using custom values for them.
    """
    def __init__(self, mdp: tm.TohMdp, alpha: float = 0.1, epsilon: float = 0.1):
        super().__init__(mdp)
        self.alpha: Optional[float] = alpha
        self.epsilon: Optional[float] = epsilon
        self.use_custom_alpha = False
        self.use_custom_epsilon = False
        self.random_seed = 0xcafebabe
        self.rng = random.Random(self.random_seed)
        self.n_update = 1

    @property
    def policy(self) -> tm.Policy:
        return solver_utils.extract_policy(self.mdp, self.q_table)

    @property
    def v_table(self) -> tm.VTable:
        return solver_utils.extract_v_table(self.mdp, self.q_table)

    def epsilon_greedy(
            self, best_actions: List[tm.TohAction],
            epsilon: float) -> tm.TohAction:
        """Sample action according to epsilon-greedy strategy.

        With probability 1-epsilon, samples one of the best_actions;
        with probability epsilon, samples a totally random action.

        Args:
            best_actions: the best actions at the current state
            epsilon: the epsilon value to use

        Returns:
            action: tm.TohAction
                the sampled action.
        """
        coin = self.rng.random()
        if coin >= epsilon:
            best_actions = sorted(best_actions)
            return self.rng.choice(best_actions)
        else:
            return self.rng.choice(self.mdp.actions[:-1])

    def choose_next_action(self, state: tm.TohState) -> tm.TohAction:
        assert state != self.mdp.terminal
        if self.mdp.is_goal(state):
            return 'Exit'

        epsilon = (solver_utils.custom_epsilon(self.n_update)
                   if self.use_custom_epsilon else self.epsilon)
        return solver_utils.choose_next_action(
            self.mdp, state, epsilon, self.q_table, self.epsilon_greedy)  # type: ignore

    def q_update(self, state: tm.TohState, action: tm.TohAction,
                 reward: float, next_state: tm.TohState) -> None:
        alpha = (solver_utils.custom_alpha(self.n_update)
                 if self.use_custom_alpha else self.alpha)
        transition = (state, action, reward, next_state)
        solver_utils.q_update(self.mdp, self.q_table, transition, alpha)  # type: ignore
        self.n_update += 1

    def reset(self) -> None:
        super().reset()
        self.n_update = 1
        self.rng = random.Random(self.random_seed)
