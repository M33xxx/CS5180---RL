import env
import env_kings
import env_kings_9step
import env_Stochastic
import algorithms
import gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange

def plot_x_y(nu, time_step):
    x = []
    y = []

    for i in range(len(nu)):
        if nu[i] < time_step:
            x.append(nu[i])
            y.append(i)
    return x, y

def Q4_b(trials: int, time_step):
    env.register_env()

    X1 = []
    X2 = []
    X3 = []
    X4 = []

    Y1 = []
    Y2 = []
    Y3 = []
    Y4 = []

    for t in trange(trials, desc="Trials"):
        nu1 = algorithms.sarsa(gym.make('WindyGridWorld-v0'), 200, 1, 0.1, 0.5)
        x1, y1 = plot_x_y(nu1, time_step)
        X1.append(x1)
        Y1.append(y1)

        nu2 = algorithms.q_learning(gym.make('WindyGridWorld-v0'), 200, 1, 0.1, 0.5)
        x2, y2 = plot_x_y(nu2, time_step)
        X2.append(x2)
        Y2.append(y2)

        nu3 = algorithms.off_policy_exp_sarsa(gym.make('WindyGridWorld-v0'), 200, 1, 0.1, 0.5)
        x3, y3 = plot_x_y(nu3, time_step)
        X3.append(x3)
        Y3.append(y3)

        nu4 = algorithms.on_policy_exp_sarsa(gym.make('WindyGridWorld-v0'), 200, 1, 0.1, 0.5)
        x4, y4 = plot_x_y(nu4, time_step)
        X4.append(x4)
        Y4.append(Y4)

        # nu5 = algorithms.on_policy_mc_control_epsilon_soft(gym.make('WindyGridWorld-v0'), 200, 1, 0.1)

    ave_Y1 = np.average(Y1, 0)
    ave_Y2 = np.average(Y2, 0)
    ave_Y3 = np.average(Y3, 0)
    ave_Y4 = np.average(Y4, 0)

    ave_X1 = np.average(X1, 0)
    ave_X2 = np.average(X2, 0)
    ave_X3 = np.average(X3, 0)
    ave_X4 = np.average(X4, 0)

    std_Y1 = np.std(Y1, 0)
    std_Y2 = np.std(Y2, 0)
    std_Y3 = np.std(Y3, 0)
    std_Y4 = np.std(Y4, 0)

    l_1 = (ave_Y1 - 1.96 * (std_Y1/np.sqrt(trials))).flatten()
    h_1 = (ave_Y1 + 1.96 * (std_Y1/np.sqrt(trials))).flatten()

    l_2 = (ave_Y2 - 1.96 * (std_Y2/np.sqrt(trials))).flatten()
    h_2 = (ave_Y2 + 1.96 * (std_Y2/np.sqrt(trials))).flatten()

    l_3 = (ave_Y3 - 1.96 * (std_Y3/np.sqrt(trials))).flatten()
    h_3 = (ave_Y3 + 1.96 * (std_Y3/np.sqrt(trials))).flatten()

    l_4 = (ave_Y4 - 1.96 * (std_Y4/np.sqrt(trials))).flatten()
    h_4 = (ave_Y4 + 1.96 * (std_Y4/np.sqrt(trials))).flatten()

    plt.fill_between(ave_X1, l_1, h_1, alpha=0.1)
    plt.fill_between(ave_X2, l_2, h_2, alpha=0.1)
    plt.fill_between(ave_X3, l_3, h_3, alpha=0.1)
    plt.fill_between(ave_X4, l_4, h_4, alpha=0.1)

    plt.figure()
    plt.plot(ave_X1, ave_Y1, label='sarsas')
    plt.plot(ave_X2, ave_Y2, label='q-learning')
    plt.plot(ave_X3, ave_Y3, label='off_policy_exp_sarsas')
    plt.plot(ave_X4, ave_Y4, label='on_policy_exp_sarsas')

    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    plt.legend()
    plt.show()

def Q4_b_1(time_step):
    env.register_env()

    nu1 = algorithms.sarsa(gym.make('WindyGridWorld-v0'), 200, 1, 0.1, 0.5)
    x1, y1 = plot_x_y(nu1, time_step)

    nu2 = algorithms.q_learning(gym.make('WindyGridWorld-v0'), 200, 1, 0.1, 0.5)
    x2, y2 = plot_x_y(nu2, time_step)

    nu3 = algorithms.off_policy_exp_sarsa(gym.make('WindyGridWorld-v0'), 200, 1, 0.1, 0.5)
    x3, y3 = plot_x_y(nu3, time_step)

    nu4 = algorithms.on_policy_exp_sarsa(gym.make('WindyGridWorld-v0'), 200, 1, 0.1, 0.5)
    x4, y4 = plot_x_y(nu4, time_step)

    # nu5 = algorithms.nstep_sarsa(gym.make('WindyGridWorld-v0'), 10, 1, 0.1, 0.5)
    # x5, y5 = plot_x_y(nu5, time_step)

    nu6 = algorithms.on_policy_mc_control_epsilon_soft(gym.make('WindyGridWorld-v0'), 200, 1, 0.1)
    x6, y6 = plot_x_y(nu6, time_step)

    plt.figure(1)
    plt.plot(x1, y1, label='sarsas')
    plt.plot(x2, y2, label='q-learning')
    plt.plot(x3, y3, label='off_policy_exp_sarsas')
    plt.plot(x4, y4, label='on_policy_exp_sarsas')
    # plt.plot(x5, y5, label='nstep_sarsa')
    plt.plot(x6, y6, label='mc_soft')

    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    plt.legend()
    plt.show()

def Q4_c(time_step):
    env_kings.register_env()

    nu1 = algorithms.sarsa(gym.make('WindyGridWorld-v0'), 200, 1, 0.1, 0.5)
    x1, y1 = plot_x_y(nu1, time_step)

    nu2 = algorithms.q_learning(gym.make('WindyGridWorld-v0'), 200, 1, 0.1, 0.5)
    x2, y2 = plot_x_y(nu2, time_step)

    nu3 = algorithms.off_policy_exp_sarsa(gym.make('WindyGridWorld-v0'), 200, 1, 0.1, 0.5)
    x3, y3 = plot_x_y(nu3, time_step)

    nu4 = algorithms.on_policy_exp_sarsa(gym.make('WindyGridWorld-v0'), 200, 1, 0.1, 0.5)
    x4, y4 = plot_x_y(nu4, time_step)

    nu6 = algorithms.on_policy_mc_control_epsilon_soft(gym.make('WindyGridWorld-v0'), 200, 1, 0.1)
    x6, y6 = plot_x_y(nu6, time_step)

    plt.figure(2)
    plt.plot(x1, y1, label='sarsas')
    plt.plot(x2, y2, label='q-learning')
    plt.plot(x3, y3, label='off_policy_exp_sarsas')
    plt.plot(x4, y4, label='on_policy_exp_sarsas')
    plt.plot(x6, y6, label='mc_soft')

    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step (Kings' move)")
    plt.legend()
    plt.show()

def Q4_c_1(time_step):
    env_kings_9step.register_env()

    nu1 = algorithms.sarsa(gym.make('WindyGridWorld-v0'), 200, 1, 0.1, 0.5)
    x1, y1 = plot_x_y(nu1, time_step)

    nu2 = algorithms.q_learning(gym.make('WindyGridWorld-v0'), 200, 1, 0.1, 0.5)
    x2, y2 = plot_x_y(nu2, time_step)

    nu3 = algorithms.off_policy_exp_sarsa(gym.make('WindyGridWorld-v0'), 200, 1, 0.1, 0.5)
    x3, y3 = plot_x_y(nu3, time_step)

    nu4 = algorithms.on_policy_exp_sarsa(gym.make('WindyGridWorld-v0'), 200, 1, 0.1, 0.5)
    x4, y4 = plot_x_y(nu4, time_step)

    nu6 = algorithms.on_policy_mc_control_epsilon_soft(gym.make('WindyGridWorld-v0'), 200, 1, 0.1)
    x6, y6 = plot_x_y(nu6, time_step)

    plt.figure(3)
    plt.plot(x1, y1, label='sarsas')
    plt.plot(x2, y2, label='q-learning')
    plt.plot(x3, y3, label='off_policy_exp_sarsas')
    plt.plot(x4, y4, label='on_policy_exp_sarsas')
    plt.plot(x6, y6, label='mc_soft')

    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step (Kings' move with 9 steps)")
    plt.legend()
    plt.show()

def Q4_d(time_step):
    env_Stochastic.register_env()

    nu1 = algorithms.sarsa(gym.make('WindyGridWorld-v0'), 200, 1, 0.1, 0.5)
    x1, y1 = plot_x_y(nu1, time_step)

    nu2 = algorithms.q_learning(gym.make('WindyGridWorld-v0'), 200, 1, 0.1, 0.5)
    x2, y2 = plot_x_y(nu2, time_step)

    nu3 = algorithms.off_policy_exp_sarsa(gym.make('WindyGridWorld-v0'), 200, 1, 0.1, 0.5)
    x3, y3 = plot_x_y(nu3, time_step)

    nu4 = algorithms.on_policy_exp_sarsa(gym.make('WindyGridWorld-v0'), 200, 1, 0.1, 0.5)
    x4, y4 = plot_x_y(nu4, time_step)

    nu6 = algorithms.on_policy_mc_control_epsilon_soft(gym.make('WindyGridWorld-v0'), 200, 1, 0.1)
    x6, y6 = plot_x_y(nu6, time_step)

    plt.figure(4)
    plt.plot(x1, y1, label='sarsas')
    plt.plot(x2, y2, label='q-learning')
    plt.plot(x3, y3, label='off_policy_exp_sarsas')
    plt.plot(x4, y4, label='on_policy_exp_sarsas')
    plt.plot(x6, y6, label='mc_soft')

    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step (Stochastic wind)")
    plt.legend()
    plt.show()

def main():

    # Q4_b(1, 8000)
    Q4_b_1(8000)
    Q4_c(8000)
    Q4_c_1(8000)
    Q4_d(8000)

if __name__ == "__main__":
    main()