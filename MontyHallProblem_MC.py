# Monte Carlo Simulation for the Monty Hall problem
import numpy as np


def monty_hall_prob(n=3, num_mc=1000):
    '''
    :param n: num of doors
    :param num_mc: num of mc iteration
    :return: prob(winning for the first choice), prob(winning with a changed mind)
    '''

    scenario = np.random.rand(num_mc, n)
    first_choice = np.random.rand(num_mc, n)
    idx_s = np.argmax(scenario, axis=1)       # locate the door with a car
    idx_c = np.argmax(first_choice, axis=1)   # record participant's first choice
    scenario[:] = 0
    first_choice[:] = 0
    for i in range(num_mc):
        scenario[i, idx_s[i]] = 1
        first_choice[i, idx_c[i]] = 1

    # compute the probability of winning for the first choice
    first_result = scenario * first_choice
    p1 = first_result.sum() / num_mc   # Attention: outputs differ while using first_result.sum() and sum(first_result)

    second_choice = np.zeros((num_mc, n))
    for i in range(num_mc):
        x = np.arange(n).tolist()
        y = np.arange(n).tolist()
        # the host chooses which door to open
        x.remove(idx_s[i])               # remove the correct one
        if sum(first_result[i]) == 1:    # participant's first choice is correct
            door_open = np.random.choice(x, 1, p=np.ones(n-1)/(n-1))[0]
        elif sum(first_result[i]) == 0:  # participant's first choice is wrong
            x.remove(idx_c[i])
            door_open = np.random.choice(x, 1, p=np.ones(n-2)/(n-2))[0]
        else:
            break

        y.remove(door_open)
        y.remove(idx_c[i])
        second_choice[i, np.random.choice(y, 1, p=np.ones(len(y))/(len(y)))[0]] = 1

    # compute the probability of winning with a changed mind
    second_result = scenario * second_choice
    p2 = second_result.sum() / num_mc

    return [p1, p2]

