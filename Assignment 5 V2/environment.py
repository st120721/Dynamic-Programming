import numpy as np


class Agent:
    def __init__(self, maze_path):
        with open(maze_path, "r") as f:
            maze = [l.replace("\n", "").split(" ") for l in f.readlines() if l[0] != "#"]
            maze.reverse()
            self.maze = np.array(maze)

    @ staticmethod
    def get_around_states(state):
        i, j = state
        around_states = {1: [(i + 1, j), (i + 1, j + 1), (i + 1, j - 1)],
                         2: [(i - 1, j), (i - 1, j + 1), (i - 1, j - 1)],
                         3: [(i, j - 1), (i + 1, j - 1), (i - 1, j - 1)],
                         4: [(i, j + 1), (i + 1, j + 1), (i - 1, j + 1)]}
        return around_states

    def get_allowed_actions(self, state):
        if self.maze[state] == "G":
            return [0]
        else:
            around_states = self.get_around_states(state)
            actions = [k for k, v in around_states.items() if self.maze[v[0]] != "1"]
            return actions

    def get_successors_and_probabilities(self, state, action):
        if action == 0:
            return {state: 1}
        else:
            p = 0.1
            around_states = self.get_around_states(state)
            adjacent_s = around_states[action][1:3]
            successors = [adjacent_s[x]
                          for x in range(2) if self.maze[adjacent_s[x]] != "1"]
            probabilities = [p] * len(successors)
            probabilities.append(1 - p * len(successors))
            successors.append(around_states[action][0])
        return dict(zip(successors, probabilities))

    def get_all_possible_states(self):
        all_states = [(x, y) for x in range(self.maze.shape[0])
                      for y in range(self.maze.shape[1]) if self.maze[x, y] != "1"]
        return all_states

    def get_successors_and_probabilities_of_all_states(self):
        all_states = self.get_all_possible_states()
        all_dict = {}
        for i, state in enumerate(all_states):
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


