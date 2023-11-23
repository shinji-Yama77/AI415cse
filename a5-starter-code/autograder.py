#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import collections
import datetime
import json
import logging
import math
import os
import random
import sys
import time
import tkinter
import traceback
import zoneinfo
from typing import List, Tuple, Callable, Dict, TypeVar

import gui
import solver_utils
import solvers
import toh_mdp as tm

logger = logging.getLogger(__name__)

T = TypeVar("T")
S = TypeVar("S")
U = TypeVar("U")
Transition = Tuple[tm.TohState, tm.TohAction, float, tm.TohState]
TESTS: List[Tuple[str, int, Callable, str]] = []
PREREQS = {}


def add_prereq(q, pre):
    if isinstance(pre, str):
        pre = [pre]

    if q not in PREREQS:
        PREREQS[q] = set()
    PREREQS[q] |= set(pre)


def test(q: str, points: int, prereqs: List[str], part: str):
    def deco(fn):
        TESTS.append((q, points, fn, part))
        add_prereq(q, prereqs)
        return fn

    return deco


class WritableNull:
    def write(self, string):
        pass

    def flush(self):
        pass


class Tracker(object):
    def __init__(self, questions, maxes, prereqs, part, use_graphics, mute_output, gs_output=False):
        self.questions = questions
        self.maxes = maxes
        self.part = part
        self.prereqs = prereqs
        self.use_graphics = use_graphics

        self.points = {q: 0 for q in self.questions}

        self.current_question = None

        self.current_test = None
        self.points_at_test_start = None
        self.possible_points_remaining = None

        self.mute_output = mute_output
        self.original_stdout = None
        self.muted = False

        self.gs_output = gs_output
        tz = zoneinfo.ZoneInfo("America/Los_Angeles")
        started_at = datetime.datetime.now(tz)
        early_bird_due = datetime.datetime(
            2021, 5, 10, hour=23, minute=59, tzinfo=tz)
        self.early_bird = started_at <= early_bird_due
        self.early_bird_bonus = 5
        print('\nStarted at %d:%02d:%02d' % started_at.timetuple()[3:6])

    def mute(self):
        if self.muted:
            return

        self.muted = True
        self.original_stdout = sys.stdout
        sys.stdout = WritableNull()

    def unmute(self):
        if not self.muted:
            return

        self.muted = False
        sys.stdout = self.original_stdout

    def begin_q(self, q):
        assert q in self.questions
        text = 'Question {}'.format(q)
        print('\n' + text)
        print('=' * len(text))

        for prereq in sorted(self.prereqs[q]):
            if prereq not in self.points or self.points[prereq] < self.maxes[prereq]:
                print("""*** NOTE: Make sure to complete Question {} before working on Question {},
*** because Question {} builds upon your answer for Question {}.""".format(prereq, q, q, prereq))
                if prereq in self.points:
                    # Cannot begin test if a prereq in current part is not
                    # satisfied.
                    return False
                else:
                    print("""*** Question {} is not in the currently graded part, but not completing 
*** Question {} may result in failing this test.
""".format(prereq, prereq))

        self.current_question = q
        self.possible_points_remaining = self.maxes[q]
        return True

    def begin_test(self, test_name):
        self.current_test = test_name
        self.points_at_test_start = self.points[self.current_question]
        print("*** {}) {}".format(self.current_question, self.current_test))
        if self.mute_output:
            self.mute()

    def end_test(self, pts):
        if self.mute_output:
            self.unmute()
        self.possible_points_remaining -= pts
        if self.points[self.current_question] == self.points_at_test_start + pts:
            print("*** PASS: {}".format(self.current_test))
        elif self.points[self.current_question] == self.points_at_test_start:
            print("*** FAIL")

        self.current_test = None
        self.points_at_test_start = None

    def end_q(self):
        assert self.current_question is not None
        assert self.possible_points_remaining == 0
        print('\n### Question {}: {}/{} ###'.format(
            self.current_question,
            self.points[self.current_question],
            self.maxes[self.current_question]))

        self.current_question = None
        self.possible_points_remaining = None

    def finalize(self):
        print('\nFinished at %d:%02d:%02d' % time.localtime()[3:6])
        print("\nProvisional grades\n==================")

        for q in self.questions:
            print('Question %s: %d/%d' % (q, self.points[q], self.maxes[q]))
        print('------------------')
        print('Total: %d/%d' % (sum(self.points.values()),
                                sum([self.maxes[q] for q in self.questions])))

        if self.gs_output:
            self.produce_gradescope_output()
        else:
            print("""
Your grades are NOT yet registered.  To register your grades, make sure
to follow your instructor's guidelines to receive credit on your assignment.
""")

    def add_points(self, pts):
        self.points[self.current_question] += pts
        if self.gs_output:
            self.produce_gradescope_output()

    def produce_gradescope_output(self):
        total_possible = sum(self.maxes.values())
        total_score = sum(self.points.values())
        gradescope_dict = {
            "score": total_score,
            "max_score": total_possible,
            "output": f"Total score ({total_score} / {total_possible})"
        }

        tests_out = []
        for name in self.questions:
            test_out = {
                "name": name,
                "score": self.points[name],
                "max_score": self.maxes[name],
                "tags": []
            }
            is_correct = self.points[name] >= self.maxes[name]
            test_out['output'] = "  Question {num} ({points}/{max}) {correct}".format(
                num=(name[1] if len(name) == 2 else name),
                points=test_out['score'],
                max=test_out['max_score'],
                correct=('X' if not is_correct else ''),
            )
            tests_out.append(test_out)

        if self.part == "p1" and self.early_bird:
            print("Good job, early bird!")
            gradescope_dict["score"] += self.early_bird_bonus
            tests_out.append({
                "name": "early_bird",
                "score": self.early_bird_bonus,
                "max_score": 0,
                "tags": [],
                "output": "  Good job, early bird!"
            })

        gradescope_dict["tests"] = tests_out

        with open("gradescope_response.json", "w") as out:
            json.dump(gradescope_dict, out)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.set_defaults(
        gs_output=False,
        no_graphics=False,
        mute_output=False,
    )
    parser.add_argument('--gradescope-output',
                        dest='gs_output',
                        action='store_true',
                        help='Generate GradeScope output files')
    parser.add_argument('--question', '-q',
                        dest='grade_question',
                        default=None,
                        choices=[q for q, _, _, _ in TESTS],
                        help='Grade only one question (e.g. `-q q1`)')
    parser.add_argument("--part", "-p",
                        dest="grade_part",
                        default=None,
                        choices=set(p for _, _, _, p in TESTS),
                        help="Grade only one part (e.g. `-p p1`)")
    parser.add_argument('--no-graphics',
                        dest='no_graphics',
                        action='store_true',
                        help='Do not display graphics (visualizing your '
                             'implementation is highly recommended for '
                             'debugging).')
    parser.add_argument('--mute',
                        dest='mute_output',
                        action='store_true',
                        help='Mute output from executing tests')
    return parser


def main():
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=os.environ.get('LOGLEVEL', 'INFO').upper(),
    )
    args = get_parser().parse_args()

    if args.gs_output:
        if args.grade_question:
            raise ValueError(
                "--gradescope-output has to be run on all questions and "
                "cannot be used together with -q/--question!")

    questions = set()
    parts = collections.defaultdict(list)
    maxes = {}
    for q, points, fn, part in TESTS:
        questions.add(q)
        parts[part].append(q)
        maxes[q] = maxes.get(q, 0) + points
        if q not in PREREQS:
            PREREQS[q] = set()

    questions = list(sorted(questions))
    if args.grade_question:
        if args.grade_question not in questions:
            print("ERROR: question {} does not exist".format(
                args.grade_question))
            sys.exit(1)
        else:
            questions = [args.grade_question]
            PREREQS[args.grade_question] = set()
    elif args.grade_part:
        if args.grade_part not in parts:
            print("ERROR: part {} does not exist".format(args.grade_part))
            sys.exit(1)
        else:
            questions = sorted(parts[args.grade_part])

    tracker = Tracker(questions, maxes, PREREQS, args.grade_part,
                      not args.no_graphics, args.mute_output, gs_output=args.gs_output)
    print(args.no_graphics)
    for q in questions:
        started = tracker.begin_q(q)
        if not started:
            continue

        for testq, points, fn, part in TESTS:
            if testq != q:
                continue
            tracker.begin_test(fn.__name__)
            # noinspection PyBroadException
            try:
                fn(tracker)
            except KeyboardInterrupt:
                tracker.unmute()
                print("\n\nCaught KeyboardInterrupt: aborting autograder")
                tracker.finalize()
                print("\n[autograder was interrupted before finishing]")
                sys.exit(1)
            except BaseException:  # Equivalent to bare except.
                tracker.unmute()
                print(traceback.format_exc())
            tracker.end_test(points)
        tracker.end_q()
    tracker.finalize()


def valmap(dict_: Dict[T, S], fn: Callable[[S], U]) -> Dict[T, U]:
    """Apply function to values of a dictionary."""
    return {k: fn(v) for k, v in dict_.items()}


def load_vi_test_case(
        n_disks, test_root="test_cases"
) -> List[Tuple[tm.TohMdpConfig, List[tm.VTable],
                List[tm.QTable], List[float]]]:
    test_case_path = os.path.join(test_root, f"vi_{n_disks}disks.json")
    with open(test_case_path, "r") as f:
        json_test_cases = json.load(f)

    test_cases = []
    for config_dict, json_v_tables, json_q_tables, max_deltas in json_test_cases:
        config = tm.TohMdpConfig(**config_dict)
        v_tables: List[tm.VTable] = [
            {tm.TohState(**valmap(state_dict, tuple)): v
             for state_dict, v in json_v_table}
            for json_v_table in json_v_tables]
        q_tables: List[tm.QTable] = [
            {(tm.TohState(**valmap(state_dict, tuple)), a): v
             for (state_dict, a), v in json_q_table}
            for json_q_table in json_q_tables]
        assert len(v_tables) == len(q_tables) == len(max_deltas), (
            f"Test case: {test_case_path} error! Please download again.")
        test_cases.append((config, v_tables, q_tables, max_deltas))

    return test_cases


@test("q1", points=30, prereqs=[], part="p1")
def check_value_iteration(tracker):
    partial_credit_per_disk = 5
    for n_disks in [2, 3, 4]:
        test_cases = load_vi_test_case(n_disks)
        for config, v_tables, q_tables, max_deltas in test_cases:
            print(f"  Testing MDP with config: {config}")
            mdp = tm.TohMdp.from_config(config)
            solver = solvers.ValueIterationSolver(mdp)
            for n_iter, (v_table, q_table, max_delta) in enumerate(
                    zip(v_tables, q_tables, max_deltas), start=1):
                got_max_delta = solver.step()
                got_v_table = solver.v_table
                got_q_table = solver.q_table

                print("    *** Checking max_delta ...", end=" ")
                assert math.isclose(max_delta, got_max_delta), (
                    f"In {n_iter}th Value Iteration, "
                    f"got max_delta={got_max_delta}, expect: {max_delta}")
                print("Passed.", end=" ")

                print("Checking v_table ...", end=" ")
                for s, v in v_table.items():
                    assert s in got_v_table, (
                        f"In {n_iter}th Value Iteration, "
                        f"{s} is not found in the solver.v_table")
                    assert math.isclose(v, got_v_table[s]), (
                        f"In {n_iter}th Value Iteration, "
                        f"got value for state {s}: {got_v_table[s]}, "
                        f"expect: {v}")
                print("Passed.", end=" ")

                print("Checking q_table calculation...", end=" ")
                for (s, a), v in q_table.items():
                    assert (s, a) in got_q_table, (
                        f"In {n_iter}th Value Iteration, "
                        f"{(s, a)} pair is not found in the solver.q_table")
                    assert math.isclose(v, got_q_table[(s, a)]), (
                        f"In {n_iter}th Value Iteration, "
                        f"got value for state-action pair {(s, a)}: "
                        f"{got_q_table[(s, a)]}, expect: {v}")
                print("Passed.")

        tracker.add_points(partial_credit_per_disk)
    tracker.add_points(15)
    if tracker.use_graphics:
        root = tkinter.Tk()
        config = tm.TohMdpConfig(0.9, 0.0, 0.2, 3, 1)
        mdp = tm.TohMdp.from_config(config)
        toh_gui = gui.GUI(root, mdp)
        toh_gui.canvas.update()
        toh_gui.vi_menu.invoke("Show state values (V) from VI")
        toh_gui.canvas.update()
        for _ in range(25):
            toh_gui.vi_menu.invoke("1 step of VI")
            time.sleep(0.2)
            toh_gui.canvas.update()
        root.destroy()


@test("q2", points=10, prereqs=[], part="p1")
def check_extract_policy(tracker):
    for n_disks in [2, 3, 4]:
        test_cases = load_vi_test_case(n_disks)
        for config, _, q_tables, _ in test_cases:
            print(f"  Testing extract_policy for MDP with config: {config}")
            mdp = tm.TohMdp.from_config(config)
            for q_table in q_tables:
                extracted_policy = solver_utils.extract_policy(mdp, q_table)
                for s in mdp.nonterminal_states:
                    assert s in extracted_policy, (
                        f"Missing state: {s} in extracted policy.")
                    policy_value = q_table[(s, extracted_policy[s])]
                    for a in mdp.actions:
                        assert policy_value >= q_table[(s, a)], (
                            f"Extracted policy for state {s} is "
                            f"{extracted_policy[s]}, but found "
                            f"q_table[{(s, a)}] = {q_table[(s, a)]}, "
                            f"which is higher than extracted policy value"
                            f"q_table[{(s, extracted_policy[s])}]: "
                            f"{policy_value}.")
    tracker.add_points(10)
    if tracker.use_graphics:
        root = tkinter.Tk()
        config = tm.TohMdpConfig(0.9, 0.0, 0.2, 3, 1)
        mdp = tm.TohMdp.from_config(config)
        toh_gui = gui.GUI(root, mdp)
        toh_gui.canvas.update()
        toh_gui.vi_menu.invoke("Show state values (V) from VI")
        toh_gui.vi_menu.invoke("Show Policy from VI")
        toh_gui.canvas.update()
        for _ in range(25):
            toh_gui.vi_menu.invoke("1 step of VI")
            time.sleep(0.5)
            toh_gui.canvas.update()
        toh_gui.vi_agent_menu.invoke("Perform 100 actions")
        root.destroy()


def load_ql_test_case(
        n_disks, test_root="test_cases"
) -> List[Tuple[tm.TohMdpConfig, List[Tuple[Transition, float, float]]]]:
    test_case_path = os.path.join(test_root, f"ql_{n_disks}disks.json")
    with open(test_case_path, "r") as f:
        json_test_cases = json.load(f)

    test_cases = []
    for config_dict, json_q_updates in json_test_cases:
        config = tm.TohMdpConfig(**config_dict)
        q_updates = []
        for json_transition, alpha, q_value in json_q_updates:
            state_dict, action, reward, next_state_dict = json_transition
            state = tm.TohState(**valmap(state_dict, tuple))
            next_state = tm.TohState(**valmap(next_state_dict, tuple))
            transition: Transition = (state, action, reward, next_state)
            q_updates.append((transition, alpha, q_value))

        test_cases.append((config, q_updates))

    return test_cases


class WatchedDict(dict):
    """Wrapped dict where only watched key can be assigned."""

    class IllegalAssignment(Exception):
        """Exception raised when unwatched key is assigned."""

        def __init__(self, key):
            super().__init__()
            self.key = key

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.watched = None

    def __setitem__(self, key, value):
        if key != self.watched:
            raise self.IllegalAssignment(key)
        else:
            super().__setitem__(key, value)


@test("q3", points=30, prereqs=[], part="p2")
def check_q_update(tracker):
    partial_credit_per_disk = 5
    for n_disks in [2, 3, 4]:
        test_cases = load_ql_test_case(n_disks)
        for config, q_updates in test_cases:
            mdp = tm.TohMdp.from_config(config)
            q_table = solvers.QLearningSolver(mdp).q_table
            q_table = WatchedDict(q_table)
            for transition, alpha, q_value in q_updates:
                state, action, reward, next_state = transition
                q_table.watched = (state, action)
                try:
                    solver_utils.q_update(mdp, q_table, transition, alpha)
                except WatchedDict.IllegalAssignment as e:
                    print(f"During handling transition: {transition}, "
                          f"another state-action pair: {e.key} is updated, "
                          f"which is unnecessary.")
                    return
                assert math.isclose(q_table[(state, action)], q_value), (
                    f"Updated Q-value for {(state, action)} is "
                    f"{q_table[(state, action)]}, expect: {q_value}")
        tracker.add_points(partial_credit_per_disk)
    tracker.add_points(15)


@test("q4", points=10, prereqs=[], part="p2")
def check_extract_v_table(tracker):
    for n_disks in [2, 3, 4]:
        test_cases = load_vi_test_case(n_disks)
        for config, v_tables, q_tables, _ in test_cases:
            print(f"  Testing extract_v_table for MDP with config: {config}")
            mdp = tm.TohMdp.from_config(config)
            for v_table, q_table in zip(v_tables, q_tables):
                extracted_v_table = solver_utils.extract_v_table(mdp, q_table)
                for s in mdp.nonterminal_states:
                    assert s in extracted_v_table, (
                        f"Missing state: {s} in extracted v_table.")
                    assert math.isclose(v_table[s], extracted_v_table[s]), (
                        f"Extracted value {extracted_v_table[s]} for state {s},"
                        f" expect {v_table[s]}.")
    tracker.add_points(10)


def train_q_learn(mdp: tm.TohMdp, solver: solvers.QLearningSolver, n_steps: int):
    """Performs Q learning updates for the specified number."""
    state = mdp.all_states[0]
    for _ in range(n_steps):
        action = solver.choose_next_action(state)
        next_state, reward = mdp.step(state, action)
        solver.q_update(state, action, reward, next_state)
        state = next_state if next_state != mdp.terminal else mdp.all_states[0]


def verify_solution_path_policy(mdp: tm.TohMdp, policy: tm.Policy):
    """Verifies that policy along the solution path is optimal."""
    for state, next_state in zip(mdp.golden_path, mdp.golden_path[1:]):
        action = policy[state]
        op = [o for o in mdp.operators if o.name == action]
        assert len(op) == 1
        op = op[0]
        assert op.pre_condition(state) and op.state_transfer(state) == next_state, (
            f"Your learned policy for state {state} on the solution path is not optimal.")
    print("  Your policy is optimal along the solution path!")


def display_ql_solver(mdp: tm.TohMdp, ql_solver: solvers.QLearningSolver, duration=5):
    """Displays the Q learning solver's Q-values and policy."""
    print("  Displaying Q learning results.")
    root = tkinter.Tk()
    toh_gui = gui.GUI(root, mdp)
    toh_gui.ql_solver = ql_solver
    toh_gui.qlearn_menu.invoke("Show Q values from QL")
    toh_gui.qlearn_menu.invoke("Show Policy from QL")
    toh_gui.mdp_rewards_menu.invoke("Show golden path (optimal solution)")
    toh_gui.canvas.update()
    time.sleep(duration)
    root.destroy()


class EpsilonGreedyChecker:
    """A mock callable to test epsilon greedy invocation."""
    class InvokedTwice(Exception):
        pass

    def __init__(self):
        self.called_arguments = None

    def __call__(self, best_actions, epsilon):
        if self.called_arguments is not None:
            raise self.InvokedTwice
        else:
            self.called_arguments = (best_actions, epsilon)


@test("q5", points=10, prereqs=["q2", "q3"], part="p2")
def check_epsilon_greedy(tracker):
    rng = random.Random(0)
    print("  Sanity checking...")
    for n_disks in [2, 3, 4]:
        test_cases = load_vi_test_case(n_disks)
        for config, _, q_tables, _ in test_cases:
            print(f"  Testing choose_next_action for MDP with config: {config}")
            mdp = tm.TohMdp.from_config(config)
            for q_table in q_tables:
                for s in mdp.nonterminal_states:
                    if mdp.is_goal(s):
                        continue
                    epsilon = rng.random()
                    eg_checker = EpsilonGreedyChecker()

                    # First check epsilon_greedy is called once and only once.
                    try:
                        solver_utils.choose_next_action(
                            mdp, s, epsilon, q_table, eg_checker)
                    except EpsilonGreedyChecker.InvokedTwice:
                        print("You cannot invoke epsilon_greedy more than once "
                              "in choose_next_action!")
                        return
                    assert eg_checker.called_arguments is not None, (
                        "You must use epsilon_greedy to sample your action in "
                        "choose_next_action!")

                    # Next check epsilon_greedy is called with the right args.
                    best_actions, called_eps = eg_checker.called_arguments
                    assert math.isclose(called_eps, epsilon), (
                        f"Passed in epsilon: {epsilon} but epsilon_greedy "
                        f"called with epsilon: {called_eps}. Please use the "
                        f"given epsilon in choose_next_action!")
                    # Check all best actions share the same Q-value, all such
                    # actions are included and that the value is indeed the
                    # best.
                    action_q_val = q_table[(s, best_actions[0])]
                    for a in best_actions:
                        assert math.isclose(q_table[(s, a)], action_q_val), (
                            f"one of the best_actions: {a} has Q value "
                            f"{q_table[(s, a)]} which is different than that "
                            f"of other actions.")
                    # also check other actions have lower q values
                    for a in mdp.actions:
                        if q_table[(s, a)] == action_q_val:
                            assert a in best_actions, (
                                f"action {a} has Q value {q_table[(s, a)]} "
                                f"which is the same as other actions in "
                                f"best_actions, but is not included.")
                        else:
                            assert action_q_val > q_table[(s, a)], (
                                f"Action {a} has a higher Q value than those "
                                f"in best_actions")
    print("  Sanity check passed.")
    tracker.add_points(5)

    print("  Running 10000 steps of Q learning with the epsilon greedy "
          "exploration...")
    # Load default MDP config.
    config = tm.TohMdpConfig(0.9, 0.0, 0.2, 3, 1)
    mdp = tm.TohMdp.from_config(config)
    solver = solvers.QLearningSolver(mdp, epsilon=0.2)

    n_steps = 10000
    train_q_learn(mdp, solver, n_steps)
    try:
        verify_solution_path_policy(mdp, solver.policy)
        tracker.add_points(5)
    finally:
        # If the test fails, still displays GUI for debugging.
        if tracker.use_graphics:
            display_ql_solver(mdp, solver)


@test("q6", points=10, prereqs=["q2", "q3", "q5"], part="p2")
def check_custom_epsilon(tracker):
    print("  Sanity checking...")
    epsilons = [solver_utils.custom_epsilon(n) for n in range(1, 1001)]
    assert len(set(epsilons)) >= 10, (
        "You custom_epsilon needs to return at least 10 different values on "
        "inputs from 1 to 1000.")
    print("  Sanity check passed.")
    tracker.add_points(5)

    print("  Running 10000 steps of Q learning with the custom epsilon greedy "
          "exploration...")
    # Load default MDP config.
    config = tm.TohMdpConfig(0.9, 0.0, 0.2, 3, 1)
    mdp = tm.TohMdp.from_config(config)
    solver = solvers.QLearningSolver(mdp)
    solver.use_custom_epsilon = True
    solver.epsilon = None

    n_steps = 10000
    train_q_learn(mdp, solver, n_steps)
    try:
        verify_solution_path_policy(mdp, solver.policy)
        tracker.add_points(5)
    finally:
        # If the test fails, still displays GUI for debugging.
        if tracker.use_graphics:
            display_ql_solver(mdp, solver)


if __name__ == '__main__':
    main()
