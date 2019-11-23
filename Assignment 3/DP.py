"""
    This module contains a cost function and dynamic programming.
"""
import numpy as np
import Environment as e

N = 10  # Horizon
all_states = e.get_all_states()  # itertools of all states
dict_next_states = e.get_next_states_of_all_states()  # a dictionary, that stores next states and probabilities for all states
value_table = np.full((len(list(all_states)), N + 1), None)  # create a matrix to store Vk

# A policy contains probably more than one action.
# It would be converted to string, since the element of numpy can not be list.
policy_table = np.full((len(list(all_states)), N + 1), "---")  # create a matrix to store policy


def cost_function(state, stage):
    """
    get the cost value from the table, if the cost value exists.
    Otherwise the cost value is calculated, which this function
    is recursive called to get the cost value in next stage from
    table.

    Args:
        state (tuple): input state
        stage (int): input stage

    Returns:
        value(float): cost value for the input state and stage
    """
    index = dict_next_states[str(state)]["index"]
    value = value_table[index, stage]
    # return value, if this value has been calculated.
    if value != None:
        return value
    else:
        # compute value for reuse
        allowed_actions = e.get_allowed_actions(state)  # allowed actions for the input state
        cost_u = []  # store cost for each action
        for u in allowed_actions:
            next_states, probabilities = dict_next_states[str(state)][str(u)]  # next states
            cost_to_go = e.compute_cost_to_go(state, u)
            expectation = sum([cost_function(next_state, stage + 1) * probabilities[i]
                               for i, next_state in enumerate(next_states)])  # recursive function
            cost_u.append((cost_to_go + expectation))

        value = min(cost_u)

        # store results in matrix
        value_table[index, stage] = value
        policy_table[index, stage] = "".join(
            [str(allowed_actions[i]) for i in range(len(allowed_actions)) if cost_u[i] == value])

        return value


def dynamic_programming():
    """
    implement dynamic programming and store the results of cost value and policy

    """
    print("-----------------------------")
    print("Start DP:")
    # K=N
    for i, state in enumerate(all_states):
        cost = e.compute_gn(state)
        value_table[i, N] = cost

    # k =0,1,...,9
    for k in range(N - 1, -1, -1):
        print("at stage", k)
        for i, state in enumerate(all_states):
            cost_function(state, k)

    # store Vk and policy
    np.save("Vk.npy", value_table)
    np.save("policy.npy", policy_table)
    np.savetxt("Vk.csv", value_table, fmt="%s", delimiter=',')
    np.savetxt("policy.csv", policy_table, fmt="%s", delimiter=',')

dynamic_programming()
