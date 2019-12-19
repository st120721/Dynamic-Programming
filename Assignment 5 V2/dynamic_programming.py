"""
    This module realizes Value Iteration(VI), Policy Iteration(PI)
    and Optimistic Policy Iteration(OPI) of Dynamic Programming.
"""
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


class DP:
    def __init__(self, environment, gamma):
        """
        construct function of class "Dynamic Programming(DP)"

        Args:
            environment (Environment): Environment based on a maze
            gamma (float): discount factor

        """
        self.environment = environment
        self.gamma = gamma
        self.threshold = 0.0001
        self.all_states = self.environment.get_all_possible_states()
        self.maze_dict = self.environment.get_successors_and_probabilities_of_all_states()
        self.functions_of_cost_to_go = dict(g1=self.environment.compute_g1,
                                            g2=self.environment.compute_g2)
        self.functions_of_iterations = dict(VI=self.value_iteration,
                                            PI=self.policy_iteration,
                                            OPI=self.optimistic_policy_iteration)

    def compute_expectation(self, state, action, v, cost_to_go):
        """
        calculate the expectation over successors for a certain state and action

        Args:
            state (tuple): input state
            action (int or list): action to be taken
            v (np.array): value function of successors
            cost_to_go (function): function of computing cost-to-go

        Returns:
            expectation(int): expectation
        """
        # choose one action if multiple actions are allowed for the state
        action = action[0] if isinstance(action, list) else action
        expectation = cost_to_go(state, action)
        expectation += self.gamma * sum([p * v[self.maze_dict[s]["index"]]
                                         for s, p in self.maze_dict[state][action].items()])
        return expectation

    def optimal_bellman_operator(self, v, cost_to_go):
        """
        optimal bellman operator

        Args:
            v (np.array): value function of successors
            cost_to_go (function): function of computing cost-to-go

        Returns:
            value_function(np.array): optimal value function
            optimal_policy(list): optimal policy

        """
        value_function, optimal_policy = [], []
        for state in self.all_states:
            allowed_actions = self.environment.get_allowed_actions(state)
            costs = [self.compute_expectation(state, u, v, cost_to_go) for u in allowed_actions]
            action = [allowed_actions[i] for i in range(len(allowed_actions)) if costs[i] == min(costs)]
            value_function.append(min(costs))
            optimal_policy.append(action)
        value_function = np.array(value_function)
        return value_function, optimal_policy

    def bellman_operator(self, policy, v, cost_to_go):
        """
        bellman operator under input policy

        Args:
            policy(list): input policy
            v (np.array): value function of successors
            cost_to_go (function): function of computing cost-to-go

        Returns:
            v(np.array): value function of input policy

        """
        v = np.array([self.compute_expectation(state, action, v, cost_to_go)
                      for state, action in zip(self.all_states, policy)])
        return v

    def value_iteration(self, cost_to_go):
        """
        Value Iteration(VI)

        Args:
            cost_to_go (function): function of computing cost-to-go

        Returns:
            v(np.array): optimal value function
            policy(list): optimal policy
        """
        # initialize all zero vector as value function
        v = np.zeros(len(self.all_states))
        while True:
            temp_v = v.copy()
            v, policy = self.optimal_bellman_operator(v, cost_to_go)

            # exit loop if the following condition is satisfied:
            # if value function of each state in k and k+1 stages is smaller than threshold
            is_under_threshold = np.abs(v - temp_v) <= self.threshold
            if np.all(is_under_threshold):
                break
        return v, policy

    def policy_iteration(self, cost_to_go):
        """
        Policy Iteration(PI)

        Args:
            cost_to_go (function): function of computing cost-to-go

        Returns:
            v(np.array): optimal value function
            policy(list): optimal policy
        """
        # initialize value function and policy
        policy = [[self.environment.get_allowed_actions(s)[0]] for s in self.all_states]
        v = np.zeros(len(self.all_states))
        while True:
            # infinite policy iteration
            # get value function under the policy
            while True:
                temp_v = v.copy()
                v = self.bellman_operator(policy, v, cost_to_go)

                # exit condition of IPE
                is_under_threshold = np.abs(v - temp_v) <= self.threshold
                if np.all(is_under_threshold):
                    break

            # policy improvement
            temp_v, policy = self.optimal_bellman_operator(v, cost_to_go)

            # exit condition of PI
            is_under_threshold = np.abs(v - temp_v) <= self.threshold
            if np.all(is_under_threshold):
                break
        return v, policy

    def optimistic_policy_iteration(self, cost_to_go, m=50):
        """
        Optimistic Policy Iteration(OPI)

        Args:
            cost_to_go (function): function of computing cost-to-go
            m (int): number of iterations (default: 50)

        Returns:
            v(np.array): optimal value function
            policy(list): optimal policy

        """
        # initialize value function with zeros
        v = np.zeros(len(self.all_states))
        while True:
            temp_v = v.copy()
            policy = self.optimal_bellman_operator(v, cost_to_go)[1]

            # m times iterations for PI
            for x in range(m):
                v = self.bellman_operator(policy, v, cost_to_go)

            # exit condition of OPI
            is_under_threshold = np.abs(v - temp_v) <= self.threshold
            if np.all(is_under_threshold):
                break
        return v, policy

    def plot(self, title, v, policy):
        """
        implement visualization for a policy and value function

        Args:
            title (str): title of plot
            v (np.array): value function for all possible states
            policy (list): list of policies for all possible states

        """
        maps = np.full(self.environment.maze.shape, np.nan)
        for state, value, action in zip(self.all_states, v, policy):
            maps[state] = value
            i, j = state
            # The policy of a state contains more than two actions.
            if len(action) > 1:
                for u in action:
                    x, y = self.environment.get_neighbour_states(state)[u][0]
                    plt.arrow(j, i, (y - j) * 0.25, (x - i) * 0.25, head_width=0.15, fc="k")
            # There is only one action in the policy of a state.
            elif action[0]:
                x, y = self.environment.get_neighbour_states(state)[action[0]][0]
                plt.arrow(j, i, (y - j) * 0.25, (x - i) * 0.25, head_width=0.15, fc="k")
            # plot point in goal state
            else:
                plt.plot(j, i, ".", c="k")
        cm = plt.get_cmap("RdBu")
        cm.set_bad(color="k")  # set wall color
        im = plt.imshow(maps, cmap=cm)
        plt.title(title, fontweight='bold')
        plt.xlabel("index j")
        plt.ylabel("index i")
        plt.gca().invert_yaxis()
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        plt.colorbar(im, cax=cax)
