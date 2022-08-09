import matplotlib.pyplot as plt
from env import BanditEnv
from tqdm import trange
from agent import EpsilonGreedy
from agent import UCB
import numpy as np

def q4(k: int, num_samples: int):
    """Q4

    Structure:
        1. Create multi-armed bandit env
        2. Pull each arm `num_samples` times and record the rewards
        3. Plot the rewards (e.g. violinplot, stripplot)

    Args:
        k (int): Number of arms in bandit environment
        num_samples (int): number of samples to take for each arm
    """

    env = BanditEnv(k=k)
    env.reset()

    # TODO

    r_total = []  # list of step rewards

    for i in range(k):  # bandits
        r_step = []

        for j in range(num_samples):
            r = env.step(i)  # reward on each step
            r_step.append(r)

        r_total.append(r_step)

    plt.xlabel('Reward Distribution')
    plt.ylabel('Action')

    plt.violinplot(r_total, showmedians=1)
    plt.show()

def q6(k: int, trials: int, steps: int):
    """Q6

    Implement epsilon greedy bandit agents with an initial estimate of 0

    Args:
        k (int): number of arms in bandit environment
        trials (int): number of trials
        steps (int): total number of steps for each trial
    """
    # TODO initialize env and agents here
    env = BanditEnv(k=k)

    a0 = EpsilonGreedy(10, 0, 0)   # agent
    a1 = EpsilonGreedy(10, 0, 0.1)
    a2 = EpsilonGreedy(10, 0, 0.01)

    agents = [a0, a1, a2]

    # Loop over trials

    R = []     # rewards
    R_o = []    # optimal rewards

    for t in trange(trials, desc="Trials"):
        # Reset environment and agents after every trial
        env.reset()
        r_trail = []
        r_o_trail = []

        for agent in agents:
            agent.reset()
            r_agent = []
            r_o_agent = []

            for i in range(steps):
                action = agent.choose_action()
                reward = env.step(action)
                agent.update(action, reward)

                if action == np.argmax(env.means):
                    r_o_agent.append(1)
                else:
                    r_o_agent.append(0)

                r_agent.append(reward)   # one agent go over steps

            r_trail.append(r_agent)         # agents go over steps, one trail
            r_o_trail.append(r_o_agent)

        R.append(r_trail)                   # trails
        R_o.append(r_o_trail)

    # TODO For each trial, perform specified number of steps for each type of agent
    # plt.plot(x, average_array[0], label='ε = 0')

    R_ave = np.average(R, 0)
    R_o_ave = np.average(R_o, 0)
    R_std = np.std(R, 0)

    y_e0 = R_ave[0]   # epsilon = 0
    y_e1 = R_ave[1]   # epsilon = 0.1
    y_e2 = R_ave[2]   # epsilon = 0.01

    y_o_e0 = R_o_ave[0]   # epsilon = 0
    y_o_e1 = R_o_ave[1]   # epsilon = 0.1
    y_o_e2 = R_o_ave[2]   # epsilon = 0.01

    y_e0_l = (R_ave[0] - 1.96 * (R_std[0]/np.sqrt(trials))).flatten()
    y_e0_h = (R_ave[0] + 1.96 * (R_std[0]/np.sqrt(trials))).flatten()

    y_e1_l = (R_ave[1] - 1.96 * (R_std[1]/np.sqrt(trials))).flatten()
    y_e1_h = (R_ave[1] + 1.96 * (R_std[1]/np.sqrt(trials))).flatten()

    y_e2_l = (R_ave[2] - 1.96 * (R_std[2]/np.sqrt(trials))).flatten()
    y_e2_h = (R_ave[2] + 1.96 * (R_std[2]/np.sqrt(trials))).flatten()

    max_r = np.max(env.means)   # upper bound line
    max_R = []
    max_R.append(max_r)

    plt.xlabel('Steps')
    plt.ylabel('Average Reward')

    plt.plot(y_e0, label='ε = 0')
    plt.plot(y_e1, label='ε = 0.1')
    plt.plot(y_e2, label='ε = 0.01')

    plt.axhline(y = np.mean(max_R), linestyle='--')

    x = np.arange(steps)

    plt.fill_between(x, y_e0_l, y_e0_h, alpha = 0.1 )
    plt.fill_between(x, y_e1_l, y_e1_h, alpha = 0.1 )
    plt.fill_between(x, y_e2_l, y_e2_h, alpha = 0.1 )

    plt.legend()

    plt.figure()
    plt.xlabel('Steps')
    plt.ylabel('Optimal Action')

    plt.plot(y_o_e0, label='ε = 0')
    plt.plot(y_o_e1, label='ε = 0.1')
    plt.plot(y_o_e2, label='ε = 0.01')

    plt.legend()
    plt.show()

def q7(k: int, trials: int, steps: int):
    """Q7

    Compare epsilon greedy bandit agents and UCB agents

    Args:
        k (int): number of arms in bandit environment
        trials (int): number of trials
        steps (int): total number of steps for each trial
    """
    # TODO initialize env and agents here
    env = BanditEnv(k=k)

    a0 = EpsilonGreedy(10, 0, 0)   # agent
    a1 = EpsilonGreedy(10, 5, 0)
    a2 = EpsilonGreedy(10, 0, 0.1)
    a3 = EpsilonGreedy(10, 5, 0.1)
    a4 = UCB(10, 0, 2, 0.1)

    agents = [a0, a1, a2, a3, a4]

    # Loop over trials

    R = []  # rewards
    R_o = []  # optimal rewards

    for t in trange(trials, desc="Trials"):
        # Reset environment and agents after every trial
        env.reset()
        r_trail = []
        r_o_trail = []

        for agent in agents:
            agent.reset()
            r_agent = []
            r_o_agent = []

            for i in range(steps):
                action = agent.choose_action()
                reward = env.step(action)
                agent.update(action, reward)

                if action == np.argmax(env.means):
                    r_o_agent.append(1)
                else:
                    r_o_agent.append(0)

                r_agent.append(reward)  # one agent go over steps

            r_trail.append(r_agent)  # agents go over steps, one trail
            r_o_trail.append(r_o_agent)

        R.append(r_trail)  # trails
        R_o.append(r_o_trail)

    # TODO For each trial, perform specified number of steps for each type of agent

    R_ave = np.average(R, 0)
    R_o_ave = np.average(R_o, 0)
    R_std = np.std(R, 0)

    y_e0 = R_ave[0]     # Q1 = 0, ε = 0
    y_e1 = R_ave[1]     # Q1 = 5, ε = 0
    y_e2 = R_ave[2]     # Q1 = 0, ε = 0.1
    y_e3 = R_ave[3]     # Q1 = 5, ε = 0.1
    y_e4 = R_ave[4]     # UCB, c = 2

    y_o_e0 = R_o_ave[0]
    y_o_e1 = R_o_ave[1]
    y_o_e2 = R_o_ave[2]
    y_o_e3 = R_o_ave[3]
    y_o_e4 = R_o_ave[4]

    y_e0_l = (R_ave[0] - 1.96 * (R_std[0] / np.sqrt(trials))).flatten()
    y_e0_h = (R_ave[0] + 1.96 * (R_std[0] / np.sqrt(trials))).flatten()

    y_e1_l = (R_ave[1] - 1.96 * (R_std[1] / np.sqrt(trials))).flatten()
    y_e1_h = (R_ave[1] + 1.96 * (R_std[1] / np.sqrt(trials))).flatten()

    y_e2_l = (R_ave[2] - 1.96 * (R_std[2] / np.sqrt(trials))).flatten()
    y_e2_h = (R_ave[2] + 1.96 * (R_std[2] / np.sqrt(trials))).flatten()

    y_e3_l = (R_ave[3] - 1.96 * (R_std[3] / np.sqrt(trials))).flatten()
    y_e3_h = (R_ave[3] + 1.96 * (R_std[3] / np.sqrt(trials))).flatten()

    y_e4_l = (R_ave[4] - 1.96 * (R_std[4] / np.sqrt(trials))).flatten()
    y_e4_h = (R_ave[4] + 1.96 * (R_std[4] / np.sqrt(trials))).flatten()

    max_R = []  # upper bound line
    max_r = np.max(env.means)
    max_R.append(max_r)

    plt.xlabel('Steps')
    plt.ylabel('Average Reward')

    plt.plot(y_e0, label='Q1 = 0, ε = 0 ')
    plt.plot(y_e1, label='Q1 = 5, ε = 0 ')
    plt.plot(y_e2, label='Q1 = 0, ε = 0.1 ')
    plt.plot(y_e3, label='Q1 = 5, ε = 0.1 ')
    plt.plot(y_e4, label='UCB, C = 2 ')

    plt.axhline(y=np.mean(max_R), linestyle='--')

    x = np.arange(steps)

    plt.fill_between(x, y_e0_l, y_e0_h, alpha=0.1)
    plt.fill_between(x, y_e1_l, y_e1_h, alpha=0.1)
    plt.fill_between(x, y_e2_l, y_e2_h, alpha=0.1)
    plt.fill_between(x, y_e3_l, y_e3_h, alpha=0.1)
    plt.fill_between(x, y_e4_l, y_e4_h, alpha=0.1)

    plt.legend()

    plt.figure()
    plt.xlabel('Steps')
    plt.ylabel('Optimal Action')

    plt.plot(y_o_e0, label='Q1 = 0, ε = 0')
    plt.plot(y_o_e1, label='Q1 = 5, ε = 0')
    plt.plot(y_o_e2, label='Q1 = 0, ε = 0.1')
    plt.plot(y_o_e3, label='Q1 = 5, ε = 0.1')
    plt.plot(y_o_e4, label='UCB, c = 2')

    plt.legend()
    plt.show()


def main():
    # TODO run code for all questions
    # q4(10, 2000)
    # q6(10,2000,1000)
    q7(10,2000,1000)

if __name__ == "__main__":
    main()
