import math
import gym
from typing import Optional
from collections import defaultdict
import numpy as np
from typing import Callable, Tuple
from tqdm import trange

def generate_episode(env: gym.Env, policy: Callable, es: bool = False, maxsteps = 1100):
    """A function to generate one episode and collect the sequence of (s, a, r) tuples

    This function will be useful for implementing the MC methods

    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        es (bool): Whether to use exploring starts or not
    """
    episode = []
    state = env.reset()
    i = 0

    while i < maxsteps:
        if es and len(episode) == 0:
            action = env.action_space.sample()
        else:
            action = policy(state)

        next_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward))
        if done:
            break
        state = next_state

        i += 1

    return episode

def create_epsilon_policy(Q: defaultdict, epsilon: float) -> Callable:
    """Creates an epsilon soft policy from Q values.

    A policy is represented as a function here because the policies are simple. More complex policies can be represented using classes.

    Args:
        Q (defaultdict): current Q-values
        epsilon (float): softness parameter
    Returns:
        get_action (Callable): Takes a state as input and outputs an action
    """
    # Get number of actions
    num_actions = len(Q[0])

    def get_action(state: Tuple) -> int:
        # TODO
        # You can reuse code from ex1
        # Make sure to break ties arbitrarily

        if np.random.random() < epsilon:
            action = np.random.randint(num_actions)
        else:
            action = np.random.choice(np.where(Q[state] == Q[state].max())[0])

        return action

    return get_action

def on_policy_mc_control_epsilon_soft(env: gym.Env, num_episodes: int, gamma: float, epsilon: float):

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))

    policy = create_epsilon_policy(Q, epsilon)
    nu_t = np.zeros(num_episodes)

    for i in trange(num_episodes, desc="Episode", leave=False):
        episode = generate_episode(env, policy)
        G = 0
        S = []

        for t in range(len(episode) - 1, -1, -1):
            s = episode[t][0]
            a = episode[t][1]
            r = episode[t][2]
            G = gamma * G + r

            # Update V and N here according to first visit MC
            # if s not in np.array(episode,dtype=object).reshape(-1,3)[:t,0]:
            if s not in S:
                N[s][a] = N[s][a] + 1
                Q[s][a] = Q[s][a] + (G - Q[s][a]) / N[s][a]
                S.append(s)

        nu_t[i] = len(episode)

    nu = np.zeros(len(nu_t))
    sum = 0
    for i in range(len(nu_t)):
        sum += nu_t[i]
        nu[i] = sum

    print(nu)
    return nu

def greedy_policy(Q, epsilon):
    n = len(Q[0])

    def prob(s):
        A = np.ones(n) * epsilon * 1/n
        A[np.argmax(Q[s])] += (1 - epsilon)

        return A
    return prob

def sarsa(env: gym.Env, num_steps: int, gamma: float, epsilon: float, step_size: float, ):
    """SARSA algorithm.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    p = greedy_policy(Q, epsilon)

    nu = np.zeros(num_steps)
    c = 0

    for i in range(num_steps):
        s = env.reset()
        a_p = p(s)
        a = np.random.choice(np.arange(len(a_p)), p=a_p)

        while True:
            n_s, r, done, _ = env.step(a)
            n_a_p = p(n_s)
            n_a = np.random.choice(np.arange(len(n_a_p)), p=n_a_p)
            Q[s][a] += step_size * (r + gamma * Q[n_s][n_a] - Q[s][a])
            c += 1

            if done:
                nu[i] = c
                break

            a = n_a
            s = n_s

    return nu


def nstep_sarsa(
        env: gym.Env,
        num_steps: int,
        gamma: float,
        epsilon: float,
        step_size: float,
):
    """N-step SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    p = greedy_policy(Q, epsilon)
    n = 4

    nu = np.zeros(num_steps)
    c = 0

    for i in range(num_steps):
        s = env.reset()
        a_p = p(s)
        a = np.random.choice(np.arange(len(a_p)), p=a_p)
        T = math.inf
        t = 0

        R = []
        S = []
        A = []

        S.append(s)
        A.append(a)
        while True:
            c += 1

            if t < T:
                n_s, r, done, _ = env.step(a)
                R.append(r)
                S.append(n_s)

                if done:
                    T = t + 1

                else:
                    n_a_p = p(n_s)
                    n_a = np.random.choice(np.arange(len(n_a_p)), p=n_a_p)
                    A.append(n_a)

            tao = t - n + 1

            if tao >= 0:
                G = 0

                for i in range(tao+1, min(tao+n, T)+1):
                    G += gamma**(i-tao-1) * R[i-1]

                if tao+n < T:
                    G += gamma**n * Q[S[tao+n], A[tao+n]]

                Q[S[tao], A[tao]] += step_size * (G - Q[S[tao], A[tao]])

            if tao == T-1:
                nu[i] = c
                break

            t += 1
        return


def off_policy_exp_sarsa(
        env: gym.Env,
        num_steps: int,
        gamma: float,
        epsilon: float,
        step_size: float,
):
    """Expected SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    n = env.action_space.n

    Q = defaultdict(lambda: np.zeros(n))
    p = greedy_policy(Q, epsilon)

    nu = np.zeros(num_steps)
    c = 0

    for i in range(num_steps):
        s = env.reset()

        while True:
            a_p = p(s)
            a = np.random.choice(np.arange(len(a_p)), p=a_p)

            n_s, r, done, _ = env.step(a)
            b_a = np.argmax(Q[n_s])
            Q[s][a] += step_size * ((r + gamma * (1 - epsilon) * Q[n_s][b_a]) + (epsilon / n) * np.sum([Q[s][a] for a in range(n)]))
            c += 1

            if done:
                nu[i] = c
                break

            s = n_s

    return nu

def on_policy_exp_sarsa(
        env: gym.Env,
        num_steps: int,
        gamma: float,
        epsilon: float,
        step_size: float,
):
    """Expected SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    n = env.action_space.n
    Q = defaultdict(lambda: np.zeros(n))
    p = greedy_policy(Q, epsilon)

    nu = np.zeros(num_steps)
    c = 0

    for i in range(num_steps):
        s = env.reset()
        a_p = p(s)
        a = np.random.choice(np.arange(len(a_p)), p=a_p)

        while True:
            n_s, r, done, _ = env.step(a)
            n_a_p = p(n_s)
            n_a = np.random.choice(np.arange(len(n_a_p)), p=n_a_p)

            Q[s][a] += step_size * ((r + gamma * np.sum([p(n_s)[a] * Q[n_s][a] for a in range(n)])) - Q[s][a])
            c += 1

            if done:
                nu[i] = c
                break

            a = n_a
            s = n_s

    return nu


def q_learning(
        env: gym.Env,
        num_steps: int,
        gamma: float,
        epsilon: float,
        step_size: float,
):
    """Q-learning

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    p = greedy_policy(Q, epsilon)

    nu = np.zeros(num_steps)
    c = 0

    for i in range(num_steps):
        s = env.reset()

        while True:
            a_p = p(s)
            a = np.random.choice(np.arange(len(a_p)), p=a_p)

            n_s, r, done, _ = env.step(a)
            b_a = np.argmax(Q[n_s])
            Q[s][a] += step_size * (r + gamma * np.max(Q[n_s][b_a]) - Q[s][a])
            c += 1

            if done:
                nu[i] = c
                break

            s = n_s

    return nu


def td_prediction(env: gym.Env, gamma: float, episodes, n=1) -> defaultdict:
    """TD Prediction

    This generic function performs TD prediction for any n >= 1. TD(0) corresponds to n=1.

    Args:
        env (gym.Env): a Gym API compatible environment
        gamma (float): Discount factor of MDP
        episodes : the evaluation episodes. Should be a sequence of (s, a, r) tuples or a dict.
        n (int): The number of steps to use for TD update. Use n=1 for TD(0).
    """
    # TODO
    pass


def learning_targets(
        V: defaultdict, gamma: float, episodes, n: Optional[int] = None
) -> np.ndarray:
    """Compute the learning targets for the given evaluation episodes.

    This generic function computes the learning targets for Monte Carlo (n=None), TD(0) (n=1), or TD(n) (n=n).

    Args:
        V (defaultdict) : A dict of state values
        gamma (float): Discount factor of MDP
        episodes : the evaluation episodes. Should be a sequence of (s, a, r) tuples or a dict.
        n (int or None): The number of steps for the learning targets. Use n=1 for TD(0), n=None for MC.
    """
    # TODO
    targets = np.zeros(len(episodes))

    pass
