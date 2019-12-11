import numpy as np
import Environment as e
import random
import matplotlib.pyplot as plt
import DP
# run Dynamic Programming to get the results
DP.dynamic_programming()

all_states = e.get_all_states()  # all possible states
Vk = np.load("Vk.npy")  # load Vk
policy = np.load("policy.npy")  # load policy

# Task 1
chance_to_complete_jobs = e.chance_to_complete_jobs
print("Chance to complete jobs:\n{}\n".format(chance_to_complete_jobs))

random_state_1 = ((2, 0, 0), (0, 0, 0), (1, 0, 0), 2)  # a state with almost empty queues
index_1 = all_states.index(random_state_1)
u1 = policy[index_1][0]
e.print_state_in_console(random_state_1)
print("optimal action: {}\n".format(u1))

random_state_2 = ((1, 2, 0), (1, 2, 1), (2, 2, 0), 1)  # a state with almost empty queues
index_2 = all_states.index(random_state_2)
u2 = policy[index_2][0]
e.print_state_in_console(random_state_2)
print("optimal action: {}\n".format(u2))


# Task 2
def plot_trajectories_data(stage, iteration, repetition):
    """
    draw the plot of min, max, mean of sum for cost-to-go of generated trajectories

    Args:
        stage (int): input stage
        iteration (int): the number of iteration in each stage
        repetition (int): how many times is a trajectory created and then computed

    """
    # lists used to store the values at each stage
    mean_costs = []
    min_costs = []
    max_costs = []
    for k in range(stage):
        costs_stage = []  # temporary list used to store sum of costs at each repetition
        for i in range(repetition):
            # get start state and action
            index_of_state = random.choice(range(len(all_states)))
            state = all_states[index_of_state]
            action = int(str(policy[index_of_state, k])[0])

            costs_repetition = []  # temporary list used to store costs of each state
            for j in range(iteration):
                print("\n-----------------------------")
                print("Iteration {}:".format(j + 1))

                cost_to_go = e.compute_cost_to_go(state, action)
                costs_repetition.append(cost_to_go)
                e.print_state_in_console(state)

                # generate noise
                noise = [1 if state[i][0] != 0 and random.random() > e.chance_to_complete_jobs[i, state[i][0] - 1]
                         else 0 for i in range(e.Q)]
                # get next state by applying the function of dynamic system
                next_state = e.system_dynamics(state, action, noise)
                print("take action {}".format(action))

                # get action of next state by checking the policy table
                index_of_next_state = all_states.index(next_state)
                next_action = int(str(policy[index_of_next_state, k])[0])

                # transform next state to start state
                state = next_state
                action = next_action
            costs_stage.append(sum(costs_repetition))
        mean_costs.append(np.mean(costs_stage))
        min_costs.append(np.min(costs_stage))
        max_costs.append(np.max(costs_stage))

    # plot
    plt.title('Investigating the Policy')
    plt.plot(range(stage), mean_costs, color='green', label='mean')
    plt.plot(range(stage), min_costs, color='blue', label='min ')
    plt.plot(range(stage), max_costs, color='red', label='max')
    plt.legend()
    plt.xlabel('stage')
    plt.ylabel('average sum of costs at each stage')
    plt.xticks(range(0, stage))
    plt.savefig('investigating_policy.png')
    plt.show()
