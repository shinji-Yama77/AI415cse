"""A GUI for the Towers of Hanoi state space.

Copyright by the University of Washington, 2021.

It supports displaying values in each state, and highlighting
any one state at a time.
Facilities are here for showing Q-states, and user interaction via
menus and a "driving console" to directly controlling an agent
solving the puzzle.
"""

import dataclasses
import logging
import os
import time
import tkinter as tk
import math
from typing import Optional

import solvers
import toh_mdp

logger = logging.getLogger(__name__)


class GUI:
    WIDTH = 650
    HEIGHT = 650
    TITLE = ("TOH World: A Markov Decision Process for the Towers of Hanoi "
             "(C) Univ. of Wash. CSE, 2021")
    MAX_VAL = 100.0
    MIN_VAL = -100.0
    DRIVING_ARROW_COLOR = "#8000AA"

    def __init__(self, master: tk.Tk, mdp: toh_mdp.TohMdp):
        self.master = master
        master.title(self.TITLE)

        self.state_coords = {}
        self.edge_lines = []  # Stores edge widgets between state nodes.
        self.state_to_circle = {}  # Stores state node circle widgets.
        self.state_to_highlight = {}  # Stores highlight widgets.
        self.toh_state_rects = []  # Rectangle widgets for displaying TOH.
        self.pi_line_bufs = [[], []]  # Stores policy display widgets.
        self.states_to_value_labels = {}  # Stores v_table display widgets.
        self.q_arcs_and_texts = {}  # Stores q_table display widgets.
        self.driving_arrows = []

        self.canvas = tk.Canvas(master, width=self.WIDTH, height=self.HEIGHT)
        self.canvas.configure(background="#ccccff")  # Blue-gray
        self.canvas.pack(fill="both", expand=True)
        # Background to make q values more visible.
        self.canvas.create_rectangle(
            0, self.HEIGHT * 0.3 - 70, self.WIDTH, self.HEIGHT, fill="#888888")

        # Create menus.
        self.menu = tk.Menu(master)
        self.mdp_rewards_menu = tk.Menu(self.menu, tearoff=0)
        self.vi_menu = tk.Menu(self.menu, tearoff=0)
        self.vi_agent_menu = tk.Menu(self.menu, tearoff=0)
        self.qlearn_menu = tk.Menu(self.menu, tearoff=0)
        # Setup GUI variables.
        self.noise_var = tk.IntVar(name="noise")
        self.n_goals_var = tk.IntVar(name="n_goals")
        self.living_reward_var = tk.IntVar(name="living_reward")
        self.golden_path_var = tk.BooleanVar(name="show_golden_path")
        self.display_values_var = tk.IntVar(name="display_values")
        self.show_vi_policy_var = tk.BooleanVar(name="show_vi_policy")
        self.show_ql_policy_var = tk.BooleanVar(name="show_ql_policy")
        self.user_console_var = tk.BooleanVar(name="user_console")
        self.gamma_var = tk.IntVar(name="gamma")
        self.alpha_var = tk.IntVar(name="alpha")
        self.epsilon_var = tk.IntVar(name="epsilon")
        self.exploration_var = tk.BooleanVar()
        # Setup menus.
        self.setup_all_menus()
        self.master.config(menu=self.menu)

        # Support for drawing arrows in 6 different directions, for displaying
        # policies.
        self.driving_arrow_directions = [
            (-math.pi * 2 * n / 6) for n in range(6)]
        self.directions_0 = [(-math.pi * 2 * n / 6) - 0.1 for n in range(6)]
        self.directions_1 = [(-math.pi * 2 * n / 6) + 0.1 for n in range(6)]

        self.driving_arrow_xys = [(math.cos(d), math.sin(d)) for d in
                                  self.driving_arrow_directions]
        self.policy_xys_0 = [(math.cos(d), math.sin(d))
                             for d in self.directions_0]
        self.policy_xys_1 = [(math.cos(d), math.sin(d))
                             for d in self.directions_1]

        # Finally, do some initialization needed for policy display and
        # displaying the user driving console.
        self.r = 20  # radius for state circles. OK when n_disks = 3.
        r_a = self.r * 1.6
        self.driving_arrow_segments = [
            (int(self.r * x), int(self.r * y), int(r_a * x), int(r_a * y))
            for (x, y) in self.driving_arrow_xys]
        self.q_text_deltas = [
            (self.r, self.r), (1.5 * self.r, 0), (self.r, -self.r),
            (-self.r, -self.r), (-1.5*self.r, 0), (-self.r, self.r)]
        self.last_dc_status = False  # For tracking console can only exit.
        self.segments_for_policy_0 = []
        self.segments_for_policy_1 = []

        # Placeholder values to be initialized with MDP
        self.divisor = 1  # Used in computing barycentric coordinates.
        self.q_text_font = None
        self.value_font = None
        self.mdp: toh_mdp.TohMdp = mdp
        self.vi_solver = solvers.ValueIterationSolver(mdp)
        self.ql_solver = solvers.QLearningSolver(mdp)
        self.golden_path_edges = None
        self.silver_path_edges = None
        self.agent_state: Optional[toh_mdp.TohState] = None
        self.load_and_draw_mdp(mdp)

    def setup_all_menus(self):
        self.setup_disk_menu()
        self.setup_noise_menu()
        self.setup_rewards_menu()
        self.setup_gamma_menu()
        self.setup_vi_menu()
        self.setup_vi_agent_menu()
        self.setup_qlearn_menu()
        self.setup_ql_param_menu()
        self.init_menu_settings()

    def setup_disk_menu(self):
        filemenu = tk.Menu(self.menu, tearoff=0)
        filemenu.add_command(label="Restart with 2 disks",
                             command=lambda: self.update_mdp_config(n_disks=2))
        filemenu.add_command(label="Restart with 3 disks",
                             command=lambda: self.update_mdp_config(n_disks=3))
        filemenu.add_command(label="Restart with 4 disks",
                             command=lambda: self.update_mdp_config(n_disks=4))
        filemenu.add_command(label="Exit", command=exit)
        self.menu.add_cascade(label="File", menu=filemenu)

    def setup_noise_menu(self):
        mdp_noise_menu = tk.Menu(self.menu, tearoff=0)
        mdp_noise_menu.add_checkbutton(
            label="0% (deterministic)", var=self.noise_var, onvalue=1,
            offvalue=2, command=lambda: self.update_mdp_config(noise=0.0))
        mdp_noise_menu.add_checkbutton(
            label="20%", var=self.noise_var, onvalue=2, offvalue=1,
            command=lambda: self.update_mdp_config(noise=0.2))
        self.menu.add_cascade(label="MDP Noise", menu=mdp_noise_menu)

    def setup_rewards_menu(self):
        self.mdp_rewards_menu.add_checkbutton(
            label="One goal, R=100", var=self.n_goals_var, onvalue=1,
            offvalue=2, command=lambda: self.update_mdp_config(n_goals=1))
        self.mdp_rewards_menu.add_checkbutton(
            label="Two goals, R=100 and R=10", var=self.n_goals_var, onvalue=2,
            offvalue=1, command=lambda: self.update_mdp_config(n_goals=2))
        self.mdp_rewards_menu.add_radiobutton(
            label="Living R= 0.0", var=self.living_reward_var, value=1,
            command=lambda: self.update_mdp_config(living_reward=0.0))
        self.mdp_rewards_menu.add_radiobutton(
            label="Living R= -0.01", var=self.living_reward_var, value=2,
            command=lambda: self.update_mdp_config(living_reward=-0.01))
        self.mdp_rewards_menu.add_radiobutton(
            label="Living R= -0.1", var=self.living_reward_var, value=3,
            command=lambda: self.update_mdp_config(living_reward=-0.1))
        self.mdp_rewards_menu.add_radiobutton(
            label="Living R= +0.1", var=self.living_reward_var, value=4,
            command=lambda: self.update_mdp_config(living_reward=0.1))
        self.mdp_rewards_menu.add_checkbutton(
            label="Show golden path (optimal solution)",
            var=self.golden_path_var, onvalue=True,
            command=self.show_golden_path)
        self.menu.add_cascade(label="MDP Rewards", menu=self.mdp_rewards_menu)

    def setup_gamma_menu(self):
        gamma_menu = tk.Menu(self.menu, tearoff=0)
        gamma_menu.add_radiobutton(
            label="\u03b3 = 1.0", var=self.gamma_var, value=1,
            command=lambda: self.update_mdp_config(gamma=1.0))
        gamma_menu.add_radiobutton(
            label="\u03b3 = 0.99", var=self.gamma_var, value=2,
            command=lambda: self.update_mdp_config(gamma=0.99))
        gamma_menu.add_radiobutton(
            label="\u03b3 = 0.9", var=self.gamma_var, value=3,
            command=lambda: self.update_mdp_config(gamma=0.9))
        gamma_menu.add_radiobutton(
            label="\u03b3 = 0.5", var=self.gamma_var, value=4,
            command=lambda: self.update_mdp_config(gamma=0.5))
        self.menu.add_cascade(label="Discount", menu=gamma_menu)

    def setup_vi_menu(self):

        # Define all callbacks.
        def update_vi_displays():
            self.update_value_displays()
            self.update_vi_policy_displays()

        def reset_vi_solver():
            self.vi_solver.reset()
            update_vi_displays()
            self.configure_value_iteration(True)
            self.configure_vi_action_menu_items(False)

        def run_value_iteration(n_steps):
            for _ in range(n_steps):
                self.vi_solver.step()
                if self.vi_solver.converged:
                    break
            self.configure_policy_extraction(True)
            update_vi_displays()

        # Add menu items.
        self.vi_menu.add_checkbutton(
            label="Show state values (V) from VI",
            var=self.display_values_var, onvalue=1,
            command=update_vi_displays)
        self.vi_menu.add_checkbutton(
            label="Show Q values from VI",
            var=self.display_values_var, onvalue=2,
            command=update_vi_displays)
        self.vi_menu.add_command(
            label="Reset state values (V) and Q values for VI to 0",
            command=reset_vi_solver)
        self.vi_menu.add_command(
            label="1 step of VI", command=lambda: run_value_iteration(1))
        self.vi_menu.add_command(
            label="10 steps of VI", command=lambda: run_value_iteration(10))
        self.vi_menu.add_command(
            label="100 steps of VI", command=lambda: run_value_iteration(100))
        self.vi_menu.add_checkbutton(
            label="Show Policy from VI",
            var=self.show_vi_policy_var, onvalue=True,
            command=self.update_vi_policy_displays)
        self.menu.add_cascade(label="Value Iteration", menu=self.vi_menu)

    def setup_vi_agent_menu(self):

        def run_vi_agent(n_steps):
            policy = self.vi_solver.policy
            if self.agent_state is None:
                self.move_agent(self.mdp.all_states[0])

            for _ in range(n_steps):
                action = policy[self.agent_state]
                next_state, reward = self.mdp.step(self.agent_state, action)
                logger.info("VI Agent: %s --- %s --> %s, reward: %s",
                            self.agent_state, action, next_state, reward)
                if not self.move_agent(next_state):
                    break

        self.vi_agent_menu.add_command(
            label="Reset state to s0",
            command=lambda: self.move_agent(self.mdp.all_states[0]))
        self.vi_agent_menu.add_command(label="Perform 1 action",
                                       command=lambda: run_vi_agent(1))
        self.vi_agent_menu.add_command(label="Perform 10 actions",
                                       command=lambda: run_vi_agent(10))
        self.vi_agent_menu.add_command(label="Perform 100 actions",
                                       command=lambda: run_vi_agent(100))
        self.menu.add_cascade(label="VI Agent", menu=self.vi_agent_menu)

    def setup_qlearn_menu(self):
        def update_ql_displays():
            self.update_value_displays()
            self.update_ql_policy_displays()

        def reset_ql_solver():
            self.ql_solver.reset()
            update_ql_displays()

        def train_qlearn_early_stop(n_steps):
            if self.agent_state is None:
                self.move_agent(self.mdp.all_states[0])

            for _ in range(n_steps):
                action = self.ql_solver.choose_next_action(self.agent_state)
                can_continue = self.qlearn_take_action(action)
                if not can_continue:
                    break

        def train_qlearn_quietly(n_steps):
            state = self.agent_state  # Stop updating agent_state.
            for _ in range(n_steps):
                if state is None or state == self.mdp.terminal:
                    state = self.mdp.all_states[0]
                action = self.ql_solver.choose_next_action(state)
                next_state, reward = self.mdp.step(state, action)
                self.ql_solver.q_update(state, action, reward, next_state)
                state = next_state

            # Restore agent_state and display.
            update_ql_displays()
            if state is None or state == self.mdp.terminal:
                self.move_agent(self.mdp.all_states[0])
            else:
                self.move_agent(state)

        self.qlearn_menu.add_checkbutton(
            label="Show state values (V) from QL",
            var=self.display_values_var, onvalue=3, command=update_ql_displays)
        self.qlearn_menu.add_checkbutton(
            label="Show Q values from QL",
            var=self.display_values_var, onvalue=4, command=update_ql_displays)
        self.qlearn_menu.add_command(
            label="Reset state values (V) and Q values for QL to 0",
            command=reset_ql_solver)
        self.qlearn_menu.add_command(
            label="Reset state to s0",
            command=lambda: self.move_agent(self.mdp.all_states[0]))
        self.qlearn_menu.add_checkbutton(
            label="User driving console",
            var=self.user_console_var, onvalue=True, offvalue=False,
            command=self.update_user_driving_console)
        self.qlearn_menu.add_command(
            label="Perform 1 action",
            command=lambda: train_qlearn_early_stop(1))
        self.qlearn_menu.add_command(
            label="Perform up to 10 actions",
            command=lambda: train_qlearn_early_stop(10))
        self.qlearn_menu.add_command(
            label="Perform up to 100 actions",
            command=lambda: train_qlearn_early_stop(100))
        self.qlearn_menu.add_command(
            label="Train for 1000 transitions",
            command=lambda: train_qlearn_quietly(1000))
        self.qlearn_menu.add_command(
            label="Train for 5000 transitions",
            command=lambda: train_qlearn_quietly(5000))
        self.qlearn_menu.add_checkbutton(
            label="Show Policy from QL",
            var=self.show_ql_policy_var, onvalue=True,
            command=self.update_ql_policy_displays)
        self.qlearn_menu.add_command(
            label="Compare results of Q-Learning and Value Iteration",
            command=lambda: NotImplementedError)
        self.qlearn_menu.entryconfig(
            "Compare results of Q-Learning and Value Iteration",
            state="disabled")
        self.menu.add_cascade(label="Q-Learning", menu=self.qlearn_menu)

    def setup_ql_param_menu(self):
        alpha_epsilon_values = {
            1: (0.1, False),
            2: (0.2, False),
            3: (None, True),
        }

        def update_alpha():
            (self.ql_solver.alpha, self.ql_solver.use_custom_alpha
             ) = alpha_epsilon_values[self.alpha_var.get()]

        def update_epsilon():
            (self.ql_solver.epsilon, self.ql_solver.use_custom_epsilon
             ) = alpha_epsilon_values[self.epsilon_var.get()]

        ql_param_menu = tk.Menu(self.menu, tearoff=0)
        ql_param_menu.add_radiobutton(
            label="Fixed \u03b1=0.1", var=self.alpha_var, value=1,
            command=update_alpha)
        ql_param_menu.add_radiobutton(
            label="Fixed \u03b1=0.2", var=self.alpha_var, value=2,
            command=update_alpha)
        ql_param_menu.add_radiobutton(
            label="Custom \u03b1", var=self.alpha_var, value=3,
            command=update_alpha)
        ql_param_menu.add_radiobutton(
            label="Fixed \u03b5=0.1", var=self.epsilon_var, value=1,
            command=update_epsilon)
        ql_param_menu.add_radiobutton(
            label="Fixed \u03b5=0.2", var=self.epsilon_var, value=2,
            command=update_epsilon)
        ql_param_menu.add_radiobutton(
            label="Custom \u03b5", var=self.epsilon_var, value=3,
            command=update_epsilon)
        ql_param_menu.add_checkbutton(
            label="Use exploration function (and reset Q values)",
            var=self.exploration_var, onvalue=True,
            command=lambda: NotImplementedError)
        ql_param_menu.entryconfig(
            "Use exploration function (and reset Q values)", state="disabled")
        self.menu.add_cascade(label="QL Params", menu=ql_param_menu)

    def init_menu_settings(self):
        self.noise_var.set(2)
        self.n_goals_var.set(1)
        self.living_reward_var.set(1)
        self.gamma_var.set(3)
        self.alpha_var.set(1)
        self.epsilon_var.set(1)
        self.configure_most_ql_menu_items(True)
        self.configure_value_iteration(True)
        self.display_values_var.set(0)

    def configure_value_iteration(self, enabled):
        begin = 1 if enabled else 3
        for i in range(begin, 6):
            self.vi_menu.entryconfig(
                i, state="normal" if enabled else "disabled")

    def configure_policy_extraction(self, enabled):
        self.vi_menu.entryconfig(
            "Show Policy from VI", state="normal" if enabled else "disabled")

    def configure_vi_action_menu_items(self, enabled):
        for i in range(1, 4):
            self.vi_agent_menu.entryconfig(
                i, state="normal" if enabled else "disabled")

    def configure_most_ql_menu_items(self, enabled):
        for i in range(1, 7):
            self.qlearn_menu.entryconfig(
                i, state="normal" if enabled else "disabled")

    def update_mdp_config(self, /, **updates):
        logger.info(
            "Updating MDP %s",
            ", ".join(f"{name}: {vars(self.mdp.config)[name]} -> {value}"
                      for name, value in updates.items()))
        config = dataclasses.replace(self.mdp.config, **updates)
        self.load_and_draw_mdp(toh_mdp.TohMdp.from_config(config))

    def move_agent(self, new_state) -> bool:
        if (self.agent_state is not None and
                self.agent_state != self.mdp.terminal):
            self.unhighlight(self.agent_state)
        self.agent_state = new_state
        can_continue = False
        if self.agent_state == self.mdp.terminal:
            logger.info("Terminal state reached.")
            self.agent_state = None
        else:
            self.highlight(self.agent_state)
            can_continue = True

        self.set_driving_console_status()
        self.canvas.update()
        time.sleep(0.5)
        return can_continue

    def qlearn_take_action(self, action):
        next_state, reward = self.mdp.step(self.agent_state, action)
        self.ql_solver.q_update(self.agent_state, action, reward, next_state)
        self.update_q_value(self.agent_state, action,
                            self.ql_solver.q_table[(self.agent_state, action)])
        return self.move_agent(next_state)

    def load_and_draw_mdp(self, mdp: toh_mdp.TohMdp):
        logger.info("Loaded MDP with %s", mdp.config)
        self.mdp = mdp
        self.vi_solver = solvers.ValueIterationSolver(mdp)
        self.ql_solver = solvers.QLearningSolver(mdp)
        self.agent_state = None

        self.clear_all_policy_displays()
        self.clear_v_q_table_displays()
        for item in self.edge_lines:
            self.canvas.delete(item)
        self.edge_lines = []
        for _, circle in self.state_to_circle.items():
            self.canvas.delete(circle)
        self.state_to_circle = {}
        for _, highlight in self.state_to_highlight.items():
            self.canvas.delete(highlight)
        self.state_to_highlight = {}

        self.r, self.q_text_font, self.value_font = {
            4: (13, ("Helvetica", 5), ("Helvetica", 9)),
            3: (20, ("Helvetica", 9), ("Helvetica", 11)),
            2: (45, ("Helvetica", 10), ("Helvetica", 14))
        }[mdp.config.n_disks]
        self.divisor = (2 ** mdp.config.n_disks) - 1

        landmarks = [(self.WIDTH * 0.1, self.HEIGHT * 0.91),
                     (self.WIDTH * 0.5, self.HEIGHT * 0.27),
                     (self.WIDTH * 0.9, self.HEIGHT * 0.91)]

        # Set coordinates for non-terminal states' nodes.
        self.state_coords = {}
        for state in mdp.state_graph:
            weights = self.barycentric(state)
            x, y = 0, 0
            for i in range(3):
                x += weights[i] * landmarks[i][0]
                y += weights[i] * landmarks[i][1]
            x, y = int(x), int(y)
            self.state_coords[state] = (x, y)

        # Draw edges between state nodes.
        self.golden_path_edges = []
        for state in mdp.state_graph:
            x0, y0 = self.state_coords[state]
            for operator, next_state in mdp.state_graph[state]:
                x1, y1 = self.state_coords[next_state]
                line = self.canvas.create_line(x0, y0, x1, y1)
                self.edge_lines.append(line)
                if (state in mdp.golden_path
                        and next_state in mdp.golden_path):
                    self.golden_path_edges.append(line)

        for state in mdp.state_graph:
            x, y = self.state_coords[state]
            self.state_to_circle[state] = self.canvas.create_oval(
                x-self.r, y-self.r, x+self.r, y+self.r, fill="yellow")

        r, r_a = self.r, int(self.r * 1.5)
        # Exit action and others go straight down.
        self.segments_for_policy_0 = [
            (int(r*x), int(r*y), int(r_a*x), int(r_a*y))
            for (x, y) in self.policy_xys_0] + [(0, r, 0, 2*r)]  # Last is exit.
        self.segments_for_policy_1 = [
            (int(r*x), int(r*y), int(r_a*x), int(r_a*y))
            for (x, y) in self.policy_xys_1] + [(int(r/5), r, int(r/5), 2*r)]

        self.display_values_var.set(0)  # Reset the display mode to yellow disks.
        self.update_toh_display(mdp.all_states[0])
        self.configure_vi_action_menu_items(False)
        self.show_vi_policy_var.set(False)
        self.show_ql_policy_var.set(False)
        self.golden_path_var.set(False)

    def show_golden_path(self) -> None:
        """Show or hide the golden path (and silver path, if n_goals=2)."""
        if self.golden_path_var.get():
            self.show_soln_path(self.golden_path_edges, "gold")
            if self.mdp.config.n_goals == 2:
                self.show_soln_path(self.silver_path_edges, "LightCyan2")
        else:
            self.show_soln_path(self.golden_path_edges, "black")
            if self.mdp.config.n_goals == 2:
                self.show_soln_path(self.silver_path_edges, "black")

    def show_soln_path(self, edges, color):
        for edge in edges:
            self.canvas.itemconfig(edge, fill=color)

    def barycentric(self, s: toh_mdp.TohState):
        """Computes all 3 barycentric coordinates for state s."""
        def make_weight(disks):
            """Computes one barycentric coord. for one peg."""
            w = 0
            for disk in disks:
                w += 2 ** (disk - 1)
            return w / self.divisor

        return [make_weight(s[p]) for p in ['peg1', 'peg2', 'peg3']]

    def show_policy(self, pi, policy_number=0, use_alt_segments=False,
                    color="brown"):
        for line in self.pi_line_bufs[policy_number]:
            self.canvas.delete(line)
        self.pi_line_bufs[policy_number] = []
        for s in pi.keys():
            a = pi[s]
            xc, yc = self.state_coords[s]
            (dx0, dy0, dx1, dy1) = self.action_to_arrow_coords(
                a, use_alt_segments=use_alt_segments)
            self.pi_line_bufs[policy_number].append(
                self.canvas.create_line(
                    xc + dx0, yc + dy0, xc + dx1, yc + dy1,
                    arrow=tk.LAST, fill=color))

    def update_vi_policy_displays(self):
        if self.show_vi_policy_var.get():
            self.show_policy(self.vi_solver.policy)
            self.configure_vi_action_menu_items(True)
        else:
            self.clear_a_policy_display(0)

    def update_ql_policy_displays(self):
        if self.show_ql_policy_var.get():
            self.show_policy(self.ql_solver.policy, policy_number=1,
                             use_alt_segments=True, color="blue")
        else:
            self.clear_a_policy_display(1)

    def clear_a_policy_display(self, policy_number):
        for line in self.pi_line_bufs[policy_number]:
            self.canvas.delete(line)
        self.pi_line_bufs[policy_number] = []

    def clear_all_policy_displays(self):
        for i in range(2):
            self.clear_a_policy_display(i)

    def display_v_table(self, v_table: toh_mdp.VTable):
        self.clear_v_q_table_displays()
        for s, v in v_table.items():
            if s != self.mdp.terminal:
                x, y = self.state_coords[s]
                label = self.canvas.create_text(
                    x, y, font=self.value_font, text=str(v))
                self.states_to_value_labels[s] = label
                self.recolor_state(s, v)  # Color the background appropriately.

    def update_value_displays(self):
        display_mode = self.display_values_var.get()
        if display_mode == 0:  # This only happens when a box is unchecked.
            self.clear_v_q_table_displays()
        elif display_mode == 1:
            self.display_v_table(self.vi_solver.v_table)
        elif display_mode == 2:
            self.display_q_table(self.vi_solver.q_table)
        elif display_mode == 3:
            self.display_v_table(self.ql_solver.v_table)
        elif display_mode == 4:
            self.display_q_table(self.ql_solver.q_table)
        else:
            logger.warning("Invalid display_values_var value!")

    def clear_v_q_table_displays(self):
        for _, label in self.states_to_value_labels.items():
            self.canvas.delete(label)
        self.states_to_value_labels = {}

        for _, (arc, text) in self.q_arcs_and_texts.items():
            self.canvas.delete(arc)
            self.canvas.delete(text)
        self.q_arcs_and_texts = {}

        for _, circle in self.state_to_circle.items():
            self.canvas.itemconfigure(circle, fill="yellow")

    def recolor_state(self, s, value, color=None) -> None:
        """Update the state's background color with the specified color; if
        the color is None, use the value to generate a color."""
        if not color:
            color = self.value_to_color(value)
        self.canvas.itemconfigure(self.state_to_circle[s], fill=color)
        label = self.states_to_value_labels[s]
        v_str = str(value)[:5] if value < 0 else str(value)[:4]
        txt_color = "white"
        self.canvas.itemconfigure(
            label, text=v_str, font=self.value_font, fill=txt_color)

    def value_to_color(self, v):
        """If v is negative return a shade of red that is brightest at -5 and
        nothing at 0.
        Otherwise, return a shade of green that is brightest at 5 and nothing
        at 0.
        The color is represented as a hex string such as #ff0000."""
        if v < 0 and v < self.MIN_VAL:
            v = self.MIN_VAL
        if v > self.MAX_VAL:
            v = self.MAX_VAL
        v /= self.MAX_VAL

        r, g, b = 0, 0, 0
        if v < 0:
            r = int(-(v * 255))
        else:
            g = int(v * 255)

        return f"#{r:02x}{g:02x}{b:02x}"

    def display_q_table(self, q_table: toh_mdp.QTable):
        """Displays the q value table.

        Display 6 sectors, color-coded by Q-values for every state in q_table
        except for terminal state. Q-value text items are sensitive to button
        clicks in case the number is illegible.
        """
        self.clear_v_q_table_displays()
        non_exit_actions = self.mdp.actions[:-1]
        arc_r = self.r
        x_scale, y_scale = 0.8, 1.
        if self.mdp.config.n_disks == 2:
            x_scale, y_scale = 1.5, 1.5
        elif self.mdp.config.n_disks == 4:
            x_scale, y_scale = 0.3, 0.5

        for s in self.mdp.nonterminal_states:
            arcs_for_s = []
            x, y = self.state_coords[s]
            for i, a in enumerate(non_exit_actions):
                q = q_table[(s, a)]
                color = self.value_to_color(q)
                arc_item = self.canvas.create_arc(
                    x - arc_r, y - arc_r, x + arc_r, y + arc_r,
                    start=(60 * i) - 30, extent=60, fill=color, outline="black")
                arcs_for_s.append(arc_item)
            # Loop separately for text, so it stays on top in all 6 sectors.
            for i, a in enumerate(non_exit_actions):
                q = q_table[(s, a)]
                q_str = " %4.3f " % q
                idx = (i + 1) % 6
                xc = x + int(x_scale * self.q_text_deltas[idx][0])
                yc = y + int(y_scale * self.q_text_deltas[idx][1])
                text_item = self.canvas.create_text(
                    xc, yc, font=self.q_text_font, text=q_str, fill="#ffffff",
                    tags=a)
                self.canvas.tag_bind(
                    text_item, '<ButtonPress-1>', self.log_q_details)
                self.q_arcs_and_texts[(s, a)] = (arcs_for_s[i], text_item)

            # add code for Exit actions to handle them specially here.
            q = q_table[(s, 'Exit')]
            q_str = " %4.3f " % q
            color = self.value_to_color(q)
            exit_r = self.r / 2
            arc_item = self.canvas.create_oval(
                x - exit_r, y - exit_r, x + exit_r, y + exit_r,
                fill=color, outline="black")
            arcs_for_s.append(arc_item)
            text_item = self.canvas.create_text(
                x, y, font=self.q_text_font, text=q_str, fill="#ffffff",
                tags="Exit")
            self.canvas.tag_bind(
                text_item, '<ButtonPress-1>', self.log_q_details)
            self.q_arcs_and_texts[(s, "Exit")] = (arc_item, text_item)
        self.configure_most_ql_menu_items(True)

    def update_q_value(self, s, a, value):
        """Change the display for this one q-value. """
        if (s, a) not in self.q_arcs_and_texts:
            logger.info("State-action pair: (%s, %s) is not on display.", s, a)
            return
        (arc_item, text_item) = self.q_arcs_and_texts[(s, a)]
        new_color = self.value_to_color(value)
        self.canvas.itemconfigure(arc_item, fill=new_color)
        q_str = " %4.3f " % value
        self.canvas.itemconfigure(text_item, text=q_str)
        logger.info("Q( %s, %s ) updated to %s", s, a, value)

    def log_q_details(self, event):
        """Callback that prints the value when a Q-value is clicked."""
        widget_id = event.widget.find_closest(event.x, event.y)[0]
        all_q_text_ids = [text for _, text in self.q_arcs_and_texts.values()]
        if widget_id in all_q_text_ids:
            logger.info("Widget ID%s clicked: Q-value = %s, Action = %s",
                        widget_id, self.canvas.itemcget(widget_id, "text"),
                        " ".join(self.canvas.gettags(widget_id)[:-1]))

    def update_toh_display(self, s: toh_mdp.TohState):
        """Update the TOH state
        This could be part of an animation."""
        for item in self.toh_state_rects:
            self.canvas.delete(item)
        self.toh_state_rects = []
        big_diam = 100  # diameter of largest disk
        big_radius = big_diam / 2
        x_center = self.WIDTH / 2
        y_base = 120
        if self.mdp.config.n_disks == 2:
            y_base = 110
        disk_height = 12
        base_width = int(big_diam * 3.5)
        base_height = disk_height
        # Draw the base for the puzzle:
        self.toh_state_rects.append(
            self.canvas.create_rectangle(
                x_center - base_width / 2, y_base,
                x_center + base_width / 2, y_base - base_height, fill="black"))

        peg_sep = int(big_diam * 1.2)
        peg_height = int(self.mdp.config.n_disks * disk_height * 1.3)
        peg_radius = 8
        x_peg = x_center - peg_sep
        for p in ['peg1', 'peg2', 'peg3']:
            # Draw a peg:
            self.toh_state_rects.append(
                self.canvas.create_rectangle(
                    x_peg - peg_radius, y_base - base_height,
                    x_peg + peg_radius, y_base - base_height - peg_height,
                    fill="brown"))
            # Draw the disks
            for i, disk in enumerate(s[p]):
                disk_radius = int(big_radius * disk / self.mdp.config.n_disks)
                self.toh_state_rects.append(self.canvas.create_rectangle(
                    x_peg-disk_radius, y_base-base_height-(i*disk_height),
                    x_peg+disk_radius, y_base-base_height-((i+1)*disk_height),
                    fill="blue"))
            x_peg += peg_sep

    def highlight(self, s: toh_mdp.TohState):
        r_h = self.r + 4
        x, y = self.state_coords[s]
        self.state_to_highlight[s] = self.canvas.create_oval(
            x - r_h, y - r_h, x + r_h, y + r_h, outline='blue', width=3)
        self.canvas.update_idletasks()
        self.update_toh_display(s)  # Draw the puzzle in this state.

    def unhighlight(self, s: toh_mdp.TohState):
        self.canvas.delete(self.state_to_highlight[s])
        del self.state_to_highlight[s]

    def action_to_arrow_coords(self, a, use_alt_segments=False):
        segments = self.segments_for_policy_0
        if use_alt_segments:
            # displaced, so can be seen with the others.
            segments = self.segments_for_policy_1
        try:
            idx = self.mdp.actions.index(a)
            return segments[idx]
        except ValueError:  # a is not found in action list.
            logger.warning("Invalid action %s when drawing policy", a)
            return segments[-1]

    def update_user_driving_console(self):
        """Display or hide 6 purple arrows on a section of the canvas for user
        to drive the agent."""
        if self.user_console_var.get():
            if not self.driving_arrows:
                xc, yc = 100, 300
                inner_scale, outer_scale = 0.5, 2
                for i in range(6):
                    (px0, py0, px1, py1) = self.driving_arrow_segments[i]
                    (x0, y0, x1, y1) = (
                        inner_scale * px0, inner_scale * py0, outer_scale * px1,
                        outer_scale * py1)
                    an_arrow = self.canvas.create_line(
                        x0 + xc, y0 + yc, x1 + xc, y1 + yc,
                        width=12, arrow=tk.LAST, fill=self.DRIVING_ARROW_COLOR,
                        tags="Action" + str(i))
                    self.canvas.tag_bind(an_arrow, '<ButtonPress-1>',
                                         self.handle_user_action_selection)
                    self.driving_arrows.append(an_arrow)
                rec = 9
                # Create a circle for the Exit action:
                exit_circle = self.canvas.create_oval(
                    xc - rec, yc - rec, xc + rec, yc + rec,
                    fill="gray", tags="Action6")
                self.canvas.tag_bind(exit_circle, '<ButtonPress-1>',
                                     self.handle_user_action_selection)
                self.driving_arrows.append(exit_circle)
        else:
            # Hide the driving console.
            for item in self.driving_arrows:
                self.canvas.delete(item)
            self.driving_arrows = []

    def set_driving_console_status(self):
        allow_exit_only = self.mdp.is_goal(self.agent_state)
        if allow_exit_only == self.last_dc_status:
            return
        self.last_dc_status = allow_exit_only
        if not self.driving_arrows:
            return
        if allow_exit_only:
            for i in range(6):
                self.canvas.itemconfigure(self.driving_arrows[i], fill="gray")
            self.canvas.itemconfigure(
                self.driving_arrows[6], fill=self.DRIVING_ARROW_COLOR)
        else:
            for i in range(6):
                self.canvas.itemconfigure(
                    self.driving_arrows[i], fill=self.DRIVING_ARROW_COLOR)
            self.canvas.itemconfigure(self.driving_arrows[6], fill="gray")

    def handle_user_action_selection(self, event):
        widget_id = event.widget.find_closest(event.x, event.y)[0]
        try:  # Clicked widget may not be an action.
            action_no = self.driving_arrows.index(widget_id)
            a = self.mdp.actions[action_no]
        except ValueError:
            logger.info("Unrecognized widget clicked.")
            return

        logger.info("Requested action is: %s", a)
        if self.last_dc_status and action_no < 6:
            logger.warning("Directional action not permitted in a goal state. "
                           "Use 'Exit' action instead.")
        else:
            if self.agent_state is None:
                self.move_agent(self.mdp.all_states[0])
            self.qlearn_take_action(a)


def main():
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=os.environ.get('LOGLEVEL', 'INFO').upper(),
    )

    root = tk.Tk()

    config = toh_mdp.TohMdpConfig(
        gamma=0.9,
        living_reward=0.0,
        noise=0.2,
        n_disks=3,
        n_goals=1,
    )
    mdp = toh_mdp.TohMdp.from_config(config)
    GUI(root, mdp)
    root.mainloop()


if __name__ == '__main__':
    main()
