import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Tuple, Callable
from enum import IntEnum


class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    STAY = 4

def actions_to_dxdy(action: Action):
    """
    Helper function to map action to changes in x and y coordinates

    Args:
        action (Action): taken action

    Returns:
        dxdy (Tuple[int, int]): Change in x and y coordinates
    """
    mapping = {
        Action.LEFT: (-1, 0),
        Action.RIGHT: (1, 0),
        Action.DOWN: (0, -1),
        Action.UP: (0, 1),
        Action.STAY: (0, 0),
    }
    return mapping[action]


def reset():
    """Return agent to start state"""
    return (0, 0)


# Q1
def simulate(state: Tuple[int, int], action: Action):
    """Simulate function for Four Rooms environment

    Implements the transition function p(next_state, reward | state, action).
    The general structure of this function is:
        1. If goal was reached, reset agent to start state
        2. Calculate the action taken from selected action (stochastic transition)
        3. Calculate the next state from the action taken (accounting for boundaries/walls)
        4. Calculate the reward

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))
        action (Action): selected action from current agent position (must be of type Action defined above)

    Returns:
        next_state (Tuple[int, int]): next agent position
        reward (float): reward for taking action in state
    """
    # Walls are listed for you
    # Coordinate system is (x, y) where x is the horizontal and y is the vertical direction
    walls = [
        (0, 5),
        (2, 5),
        (3, 5),
        (4, 5),
        (5, 0),
        (5, 2),
        (5, 3),
        (5, 4),
        (5, 5),
        (5, 6),
        (5, 7),
        (5, 9),
        (5, 10),
        (6, 4),
        (7, 4),
        (9, 4),
        (10, 4),
    ]

    # surrounding walls
    for i in range(11):
        a = [
            (-1, i),
            (i, -1),
            (11, i),
            (i, 11)
        ]
        walls += a

    # TODO check if goal was reached
    goal_state = (10, 10)
    reward = 0

    if state == goal_state:
        reward = 1
        state = reset()
        return state, reward

    # TODO modify action_taken so that 10% of the time, the action_taken is perpendicular to action (there are 2 perpendicular actions for each action)

    # 80%(80 zeros) + 10%(10 ones) + 10%(10 twos) = 100%(in a empty list)
    list1 = []
    for x in range(80):
        list1.append(0)
    for y in range(10):
        list1.append(1)
    for z in range(10):
        list1.append(2)

        a = random.choice(list1)    # probability

        b = action + 1
        if b >= 4:
            b = b - 4   # loop

        c = action + 3
        if c >= 4:
            c = c - 4   # loop

        if a == 1:
            action_taken = Action(b)
        elif a == 2:
            action_taken = Action(c)
        else:
            action_taken = action



    # TODO calculate the next state and reward given state and action_taken
    # You can use actions_to_dxdy() to calculate the next state
    # Check that the next state is within boundaries and is not a wall
    # One possible way to work with boundaries is to add a boundary wall around environment and
    # simply check whether the next state is a wall

    next_state = tuple(map(lambda i, j: i + j, actions_to_dxdy(action_taken), state))

    if (next_state in walls):
        next_state = state

    return next_state, reward


# Q2
def manual_policy(state: Tuple[int, int]):
    """A manual policy that queries user for action and returns that action

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    # TODO

    print('Please choose a new direction starting from: ' + str(state) + '; (left:0,right:2,up:3,down:1)')

    direction = input()
    move = [0, 1, 2, 3]

    if (int(direction) in move):
        action = Action(int(direction))
    else:
        action = Action(4)  # stay
        print('Please choose number from 0-3')

    return action

# Q2
def agent(
    steps: int = 1000,
    trials: int = 1,
    policy=Callable[[Tuple[int, int]], Action],
):
    """
    An agent that provides actions to the environment (actions are determined by policy), and receives
    next_state and reward from the environment

    The general structure of this function is:
        1. Loop over the number of trials
        2. Loop over total number of steps
        3. While t < steps
            - Get action from policy
            - Take a step in the environment using simulate()
            - Keep track of the reward
        4. Compute cumulative reward of trial

    Args:
        steps (int): steps
        trials (int): trials
        policy: a function that represents the current policy. Agent follows policy for interacting with environment.
            (e.g. policy=manual_policy, policy=random_policy)

    """
    # TODO you can use the following structure and add to it as needed
    R = []      # total rewards for all trails
    for t in range(trials):
        state = reset()
        i = 0

        r1 = 0      # total rewards
        r2 = []     # total rewards per trails

        while i < steps:
            # TODO select action to take
            action = policy(state)
            # TODO take step in environment using simulate()
            state, reward = simulate(state, action)
            # TODO record the reward
            r1 += reward
            r2.append(r1)
            i += 1

        R.append(r2)

    return R


# Q3
def random_policy(state: Tuple[int, int]):
    """A random policy that returns an action uniformly at random

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    # TODO
    R = [0, 1, 2, 3]    # Random choice
    from random import choice
    action = Action(choice(R))

    return action

# Q4
def worse_policy(state: Tuple[int, int]):
    """A policy that is worse than the random_policy

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    # TODO
    R = [0, 1, 2]    # Random choice except up(3) so it will never go up
    from random import choice
    action = Action(choice(R))

    return action


# Q4
def better_policy(state: Tuple[int, int]):
    """A policy that is better than the random_policy

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    # TODO
    R = [0, 1, 2, 3]

    for j in range (8):
        v = [(1, j), (8, j)]  # vertical
        h = [(j, 1), (j, 8)]  # horizontal

    for k in range(5, 11):
        rectangle = [(6, k), (7, k), (8, k), (9, k), (10, k)]   # rectangle at the top right of the map

    for p in range(2, 5):
        square = [(2, p), (3, p), (4, p)]  # square in the middle of the map

    if (state in v):
        action = Action (3)

    elif (state in h):
        action = Action (2)

    elif (state in rectangle):
        action = Action(random.choice([2, 3]))

    elif (state in square):
        action = Action(random.choice([0, 1]))

    else:
        action = Action(random.choice(R))

    return action



def main():
    # TODO run code for Q2~Q4 and plot results
    # You may be able to reuse the agent() function for each question

    # Q3 PLOT
    R_r1 = agent(10000, 10, random_policy)  # Rewards collected using random policy

    for n in range(10):
        plt.plot(R_r1[n], ':')     # random policy

        # Average rewards
    plt.plot(np.average(np.array(R_r1), 0), color='blue', label='Random Policy', linewidth=2)
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.show()

    # Q4 PLOT
    R_b = agent(10000, 10, better_policy)   # Rewards collected using better policy
    R_r = agent(10000, 10, random_policy)
    R_w = agent(10000, 10, worse_policy)

    for m in range(10):
        plt.plot(R_b[m], '--')    # better policy
        plt.plot(R_r[m], ':')     # random policy
        plt.plot(R_w[m], '_')     # worse policy

    # Average rewards
    plt.plot(np.average(np.array(R_b), 0), color='red', label='Better Policy', linewidth=2)
    plt.plot(np.average(np.array(R_r), 0), color='blue', label='Random Policy', linewidth=2)
    plt.plot(np.average(np.array(R_w), 0), color='green', label='Worse Policy', linewidth=2)

    plt.xlabel('Steps')
    plt.ylabel('Cumulative reward')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
