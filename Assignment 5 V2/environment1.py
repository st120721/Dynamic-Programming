"""
    This module contains functions of environment.
"""
import numpy as np


class Environment:
    def __init__(self, maze_path):
        """
        construct function of class "Environment"

        Args:
            maze_path (str): path of maze file

        """
        try:
            f = open(maze_path)
        except IOError:
            print("No maze file.")
        else:
            with f:
                maze = [l.replace("\n", "").split(" ") for l in f.readlines() if l[0] != "#"]
                maze.reverse()
                self.maze = np.array(maze)

    @staticmethod
    def get_around_states(state):
        """
        get around states of input state in different direction
        Each direction has 3 around states.

        Args:
            state (tuple): input state

        Returns:
            around_states(dict): a dictionary of around states

        """
        i, j = state
        around_states = {1: [(i + 1, j), (i + 1, j + 1), (i + 1, j - 1)],  # up
                         2: [(i - 1, j), (i - 1, j + 1), (i - 1, j - 1)],  # down
                         3: [(i, j - 1), (i + 1, j - 1), (i - 1, j - 1)],  # left
                         4: [(i, j + 1), (i + 1, j + 1), (i - 1, j + 1)]}  # right
        return around_states

    def get_allowed_actions(self, state):
        """
        get a list of allowed actions for a state

        Args:
            state (tuple): input state

        Returns:
            actions(list): a list of allowed actions for the input state

        """
        if self.maze[state] == "G":  # self loop in the goal
            return [0]
        else:
            actions = [k for k, v in self.get_around_states(state).items() if self.maze[v[0]] != "1"]
            return actions

    def get_all_possible_states(self):
        """
        get the states, those are not wall

        Returns:
            all_states(list): list of all possible states

        """
        all_states = [(x, y) for x in range(self.maze.shape[0])
                      for y in range(self.maze.shape[1]) if self.maze[x, y] != "1"]
        return all_states

    def get_successors_and_probabilities(self, state, action):
        """
            generate all successors and their probabilities for a certain state and action

        Args:
            state (tuple): input state
            action (int): action to be taken

        Returns:
            a dictionary of successors and their corresponding probabilities

        """
        if action == 0:
            return {state: 1}
        else:
            p = 0.1
            around_states = self.get_around_states(state)
            adjacent_s = around_states[action][1:3]
            successors = [adjacent_s[x] for x in range(2) if self.maze[adjacent_s[x]] != "1"]
            probabilities = [p] * len(successors)
            successors.append(around_states[action][0])
            probabilities.append(1 - p * len(successors))
        return dict(zip(successors, probabilities))

    def get_successors_and_probabilities_of_all_states(self):
        """
            generate all successors and their probabilities for all possible s and action

        Returns:
            a dictionary of successors and their corresponding probabilities for all states

        """
        all_dict = {}
        for i, state in enumerate(self.get_all_possible_states()):
            all_dict[state] = {}
            all_dict[state]["index"] = i
            for action in self.get_allowed_actions(state):
                all_dict[state][action] = self.get_successors_and_probabilities(state, action)
        return all_dict

    def compute_g1(self, state, action):
        if self.maze[state] == "G":
            return -1
        around_states = self.get_around_states(state)
        next_state = self.maze[around_states[action][0]]
        if next_state == "T":
            return 50
        else:
            return 0

    def compute_g2(self, state, action):
        g1_to_g2 = {-1: 0,
                    50: 50,
                    0: 1}
        cost_g1 = self.compute_g1(state, action)
        return g1_to_g2[cost_g1]
