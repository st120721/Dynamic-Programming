import environment as e
import numpy as np
import matplotlib.pyplot as plt


class DP:
    def __init__(self, agent, gamma=0.9,threshold=0.001):
        self.agent = agent
        self.gamma = gamma
        self.threshold = threshold
        self.all_states = self.agent.get_all_possible_states()
        self.all_successors = self.agent.get_successors_and_probabilities_of_all_states()
        self.functions_of_cost_to_go = [self.agent.compute_g1, self.agent.compute_g2]

    def compute_expectation(self, state, action, v, cost_to_go):
        expectation = cost_to_go(state, action)
        expectation += self.gamma * sum([p * v[self.all_successors[s]["index"]]
                                         for s, p in self.all_successors[state][action].items()])
        return expectation

    def optimal_bellman_operator(self, v, cost_to_go):
        value_function, optimal_policy = [], []
        for state in self.all_states:
            allowed_actions = self.agent.get_allowed_actions(state)
            costs = [self.compute_expectation(state, u, v, cost_to_go) for u in allowed_actions]
            value_function.append(min(costs))
            optimal_policy.append(allowed_actions[np.argmin(costs)])
        value_function = np.array(value_function)
        return value_function, optimal_policy

    def bellman_operator(self, policy, v, cost_to_go):
        v = np.array([self.compute_expectation(state, action, v, cost_to_go)
                      for state, action in zip(self.all_states, policy)])
        return v

    def value_iteration(self, cost_to_go):
        v = np.zeros(len(self.all_states))
        while True:
            temp_v = v.copy()
            v, policy = self.optimal_bellman_operator(v, cost_to_go)

            is_under_threshold = np.abs(v - temp_v) <= self.threshold
            if np.all(is_under_threshold):
                break
        return v, policy

    def policy_iteration(self, cost_to_go):
        policy = [self.agent.get_allowed_actions(s)[0] for s in self.all_states]
        v = np.zeros(len(self.all_states))

        while True:
            # ipe
            while True:
                temp_v = v.copy()
                v = self.bellman_operator(policy, v, cost_to_go)

                is_under_threshold = np.abs(v - temp_v) <= self.threshold
                if np.all(is_under_threshold):
                    break

            # policy improvement
            temp_v, policy = self.optimal_bellman_operator(v, cost_to_go)

            is_under_threshold = np.abs(v - temp_v) <= self.threshold
            if np.all(is_under_threshold):
                break
        return v, policy

    def optimistic_policy_iteration(self, cost_to_go):
        m = 50
        v = np.zeros(len(self.all_states))

        while True:
            temp_v = v.copy()
            policy = self.optimal_bellman_operator(v, cost_to_go)[1]

            for x in range(m):
                v = self.bellman_operator(policy, v, cost_to_go)

            is_under_threshold = np.abs(v - temp_v) <= self.threshold
            if np.all(is_under_threshold):
                break
        return v, policy

    def plot(self, v, policy):
        maps = np.full(self.agent.maze.shape, np.nan)
        for state, value, action in zip(self.all_states, v, policy):
            maps[state] = value
            i, j = state
            if action:
                x, y = self.agent.get_around_states(state)[action][0]
                plt.arrow(j, i, (y - j) * 0.25, (x - i) * 0.25, head_width=0.2, fc="k")
            else:
                plt.plot(j, i, ".", c="k")
        cm = plt.get_cmap("RdBu")
        cm.set_bad(color="k")
        plt.imshow(maps, cmap=cm)
        plt.colorbar()
        plt.xlabel("index j")
        plt.ylabel("index i")
        plt.gca().invert_yaxis()
        plt.show()


path = "maze2.txt"
A = e.Agent(path)
dp = DP(A, 0.9)
all = dp.all_states

maze = A.maze
# v, a = dp.value_iteration(dp.functions_of_cost_to_go[0])
policy1 = np.array([A.get_allowed_actions(s)[0] for s in all])
print(A.get_allowed_actions((3, 3)))

c = dp.functions_of_cost_to_go[0]
v1, a2 = dp.optimistic_policy_iteration(c)

dp.plot(v1, a2)
