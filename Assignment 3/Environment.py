"""
    This module contains variables and functions of environment.
"""
import random
import numpy as np
import itertools

# initialize hyperparameters
Q = 3  # number of queues
S = 3  # size of each queue
T = 2  # the total amount of job types
type_of_jobs = list(range(T + 1))  # list of type of jobs [0,1,...,T]


# State space
def print_state_in_console(state):
    """
    print the state in the console

    Args:
        state (tuple): the state to print

    """
    print("New Job to process: {}".format(state[Q]))
    for i in range(Q):
        jobs_in_ith_queue = " ".join(("-" if x == 0 else str(x) for x in state[i]))
        print("{}: {}".format(i + 1, jobs_in_ith_queue))


def push_job_into_queue(state, action):
    """
    append the job to a queue and then create a new job

    Args:
        state (tuple): input state
        action (int): input action

    Returns:
        state (tuple): the state after pushing a job

    """
    job_to_process = state[Q]
    temp_queue = list(state[action - 1])
    temp_queue[S - temp_queue.count(0)] = job_to_process
    # choose a new job to process
    new_job = random.choice(type_of_jobs)
    # insert the temporary queue into the tuple of state
    state = state[0:action - 1] + (tuple(temp_queue),) + state[action:-1] + (new_job,)
    return state


def remove_job_in_queue(state, index_of_queue):
    """
    remove the first job in a queue

    Args:
        state (tuple): input state
        index_of_queue (int): index of the queue

    Returns:
        state (tuple): the state after removing the job in the required queue

    """

    # The job can be removed, only if the queue is not empty.
    if state[index_of_queue - 1].count(0) != S:
        temp_queue = list(state[index_of_queue - 1])
        # move the (i+1)th element into ith index
        temp_queue = temp_queue[1:S] + [0]
        # insert the temporary queue into the tuple of state
        state = state[0:index_of_queue - 1] + (tuple(temp_queue),) + state[index_of_queue:]
        return state


# Action space
def get_allowed_actions(state):
    """
    get a list of allowed actions for a state

    Args:
        state (tuple): input state

    Returns:
        allowed_actions(list): a list of allowed actions for the input state

    """
    # idle, if there is no job to process
    if state[Q] == 0:
        return [0]
    else:
        # Allowed actions always include 0.
        # A queue can be chosen, if the queue is not full.
        allowed_actions = [0] + [i + 1 for i in range(Q) if state[i].count(0) > 0]
        return allowed_actions


# System Dynamics
# create a matrix of random probabilities, which represents
# the chance to finish the first job for each queue and each type of jobs
# The [i,j]element of matrix denotes the chance to complete the jth job for the ith queue.
np.random.seed(0)  # set a random seed
chance_to_complete_jobs = np.random.random(Q * T).reshape(Q, T)


def system_dynamics(state, action, noise):
    """
    implement system dynamic

    Args:
        state (tuple): input state
        action (int): input action.
        noise (list): Q-length list.
                      If the ith element of the list equals to one,
                      the first job in the corresponding queue would be finished.

    Returns:
         next_state(tuple): next state of input state

    """
    # remove the first job of a queue, if the noise of the queue equals one
    for i in range(Q):
        if noise[i]:
            state = remove_job_in_queue(state, i + 1)

    # The states stay unchanged, when they satisfy the two following conditions:
    # 1. action equals zero or the queue is full
    # 2. new job to process is not placeholder
    # i.e. ((0,0,0),(0,0,0),(0,0,0),1), action = 0
    #      ((1,1,1),(2,1,2),(1,1,2),1), action = 2
    if (action == 0 or state[action - 1].count(0) == 0) and state[-1] != 0:
        return state

    # The states get a new job to process, when the current new job to process is placeholder.
    # i.e. ((1,2,0),(1,2,0),(0,0,0),0), action = 1
    elif state[-1] == 0:
        new_job = random.choice(type_of_jobs)
        next_state = state[0:-1] + (new_job,)
        return next_state

    # Otherwise the job to process would be appended in a queue.
    else:
        next_state = push_job_into_queue(state, action)
        return next_state


def get_possible_next_states(state, action):
    """
     generate all next possible states and their probabilities for a certain state and action

    Args:
        state (tuple): input state
        action (int): action to be taken

    Returns:
        states_after_u(list): list of next states
        joint_probabilities(list): list of the corresponding probabilities of next states

    """
    queue = []  # store possibilities for each queue
    queue_probability = []  # store the probabilities for possible queues
    for i in range(Q):
        first_job = state[i][0]

        # If first job in a queue equals zero, there is only one possibility (0,0,0) with the probability of 1.
        # i.e. (0,0,0) --> (0,0,0)
        if first_job == 0:
            queue.append((state[i],))
            queue_probability.append((1,))
        else:
            # If the type of first job is 1 or 2, this job could be either completed or not.
            # i.e. (1,2,0) --> (2,0,0) or (1,2,0) --> (1,2,0)
            queue.append((remove_job_in_queue(state, i + 1)[i], state[i]))
            queue_probability.append(
                (1 - chance_to_complete_jobs[i, first_job - 1], chance_to_complete_jobs[i, first_job - 1]))

    # get all possible states and the corresponding probabilities before taking actions
    queue.append((state[Q],))
    states = itertools.product(*queue)
    states_probability = list(itertools.product(*queue_probability))

    # take actions
    states_after_u = []  # store states after taking action
    states_after_u_probability = []  # store the probabilities of states after taking action
    for index, state_ in enumerate(states):

        # The states stay unchanged, when they satisfy the two following conditions:
        # 1. Action equals zero or the queue is full.
        # 2. New job to process is not placeholder.
        # i.e. ((0,0,0),(0,0,0),(0,0,0),1), action = 0
        #      ((1,1,1),(2,1,2),(1,1,2),1), action = 2
        if (action == 0 or state_[action - 1].count(0) == 0) and state_[Q] != 0:
            states_after_u.append(state_)
            states_after_u_probability.append(states_probability[index])

        # The states get a new job to process,
        # when the current new job to process is placeholder.
        # i.e. ((1,2,0),(1,2,0),(0,0,0),0), action = 1
        elif state_[Q] == 0:
            # T+1 kinds of new job to process, each has the probability of 1/(T+1) to be chosen.
            for job in type_of_jobs:
                states_after_u.append(state_[:Q] + (job,))
                states_after_u_probability.append(states_probability[index] + (1 / (T + 1),))

        # Otherwise the job to process would be appended in a queue.
        else:
            state_ = push_job_into_queue(state_, action)
            # T+1 kinds of new job to process, each has the probability of 1/(T+1) to be chosen.
            for job in type_of_jobs:
                states_after_u.append(state_[:Q] + (job,))
                states_after_u_probability.append(states_probability[index] + (1 / (T + 1),))

    # compute joint probabilities
    joint_probabilities = []
    for x in states_after_u_probability:
        product = 1
        for y in x:
            product *= y
        joint_probabilities.append(product)

    return states_after_u, joint_probabilities


def get_all_states():
    """
    generate all possible states

    Returns:
        all_states(list): list of all possible states for an environment

    """
    # all possible queues without the restriction,
    # that there are no placeholders between jobs
    temp_queue = list(itertools.product(type_of_jobs, repeat=S))

    # A queue is inadmissible, when there is a placeholder in between two jobs.
    inadmissible_queue = [x for x in temp_queue for y in range(len(x) - 1)
                          if (x[y] == 0 and x[y + 1] != 0)]

    # remove inadmissible possibilities
    for x in inadmissible_queue:
        temp_queue.remove(x)

    all_queues = [temp_queue] * Q  # A state has Q queues.
    all_queues.append(type_of_jobs)  # T+1 type of jobs
    all_states = list(itertools.product(*all_queues))
    return all_states


def get_next_states_of_all_states():
    """
    create a dictionary, that contains the next states and their probabilities for all states.

    Returns:
        next_states_and_probabilities(dict): the created dictionary

    """
    next_states_and_probabilities = {}
    for i, state in enumerate(get_all_states()):
        next_states_and_probabilities[str(state)] = {}
        next_states_and_probabilities[str(state)]["index"] = i
        for action in get_allowed_actions(state):
            next_states_and_probabilities[str(state)][str(action)] = get_possible_next_states(state, action)
    return next_states_and_probabilities


# Local cost-to-go
def compute_cost_to_go(state, action):
    """
     compute cost-to-go for a state and action

    Args:
        state (tuple): input state
        action (int): input action

    Returns:
        cost_to_go(int): cost-to-go of the input state and action

    """
    # cost-to-go contains two parts
    cost1 = Q * S - sum([state[i].count(0) for i in range(Q)])
    cost2 = 5 if (state[Q] != 0 and action == 0) else 0
    cost_to_go = cost1 + cost2
    return cost_to_go


# Terminal cost
def compute_gn(state):
    """
    compute terminal cost of a state at the end of horizon

    Args:
        state (tuple):

    Returns:
        g_n(int): terminal cost

    """
    g_n = Q * S - sum([state[i].count(0) for i in range(Q)])
    return g_n
