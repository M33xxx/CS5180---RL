import gym
import policy
import algorithms
import env
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange


def Q3_a():
    V = algorithms.on_policy_mc_evaluation(gym.make("Blackjack-v1"), policy.default_blackjack_policy, 500000, 1)
    k = V.keys()

    u_A = np.zeros([21, 10])
    n_A = np.zeros([21, 10])

    for i in k:
        if i[2]:
            u_A[i[0] - 1, i[1] - 1] = V[i]
        else:
            n_A[i[0] - 1, i[1] - 1] = V[i]

    x1 = np.linspace(1, 10, num=10)
    y1 = np.linspace(12, 20, num=9)
    x1, y1 = np.meshgrid(x1, y1)
    z1 = u_A[12:]

    fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
    surf1 = ax1.plot_surface(x1, y1, z1, rstride=1, cstride=1, cmap='summer', linewidth=0, antialiased=False)
    fig1.colorbar(surf1, shrink=0.5, aspect=7)
    ax1.set_zlim(-1.0, 1.0)
    plt.xlabel('dealer showing')
    plt.ylabel('player sum')
    plt.title('Usable Ace after 500000 episodes')
    plt.show()

    x2 = np.linspace(1, 10, num=10)
    y2 = np.linspace(4, 20, num=17)
    x2, y2 = np.meshgrid(x2, y2)
    z2 = n_A[4:]

    fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
    surf2 = ax2.plot_surface(x2, y2, z2, rstride=1, cstride=1, cmap='summer', linewidth=0, antialiased=False)
    fig2.colorbar(surf2, shrink=0.5, aspect=7)
    ax2.set_zlim(-1.0, 1.0)
    plt.xlabel('dealer showing')
    plt.ylabel('player sum')
    plt.title('No usable Ace after 500000 episodes')
    plt.show()


def Q3_b():
    Q, policy = algorithms.on_policy_mc_control_es(gym.make("Blackjack-v1"), 5000000, 1)
    u_A = np.zeros([21, 10])
    n_A = np.zeros([21, 10])
    u_P = np.zeros([21, 10])
    n_P = np.zeros([21, 10])

    for k in Q.keys():
        if k[2]:
            u_A[k[0] - 1, k[1] - 1] = np.max(Q[k])
            u_P[k[0] - 1, k[1] - 1] = np.argmax(Q[k])

        else:
            n_A[k[0] - 1, k[1] - 1] = np.max(Q[k])
            n_P[k[0] - 1, k[1] - 1] = np.argmax(Q[k])

    plt.imshow(np.flip(u_P[11:]), cmap='summer', extent=[1, 10, 11, 21])
    plt.xlabel('dealer showing')
    plt.ylabel('player sum')
    plt.title('Policy with usable Ace')
    plt.show()

    plt.imshow(np.flip(n_P[10:]), cmap='summer', extent=[1, 10, 11, 21])
    plt.xlabel('dealer showing')
    plt.ylabel('player sum')
    plt.title('Policy with no usable Ace')
    plt.show()

    x1 = np.linspace(1, 10, num=10)
    y1 = np.linspace(12, 20, num=9)
    x1, y1 = np.meshgrid(x1, y1)
    z1 = u_A[12:]

    fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
    surf1 = ax1.plot_surface(x1, y1, z1, rstride=1, cstride=1, cmap='summer', linewidth=0, antialiased=False)
    fig1.colorbar(surf1, shrink=0.5, aspect=7)
    plt.xlabel('dealer showing')
    plt.ylabel('player sum')
    plt.title('Usable Ace after 5000000 episodes')
    plt.show()

    x2 = np.linspace(1, 10, num=10)
    y2 = np.linspace(4, 20, num=17)
    x2, y2 = np.meshgrid(x2, y2)
    z2 = n_A[4:]

    fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
    surf2 = ax2.plot_surface(x2, y2, z2, rstride=1, cstride=1, cmap='summer', linewidth=0, antialiased=False)
    fig2.colorbar(surf2, shrink=0.5, aspect=7)
    plt.xlabel('dealer showing')
    plt.ylabel('player sum')
    plt.title('No usable Ace after 5000000 episodes')
    plt.show()


def Q4_a():
    env.register_env()
    returns = algorithms.on_policy_mc_control_epsilon_soft(gym.make('FourRooms-v0'), 1000, 0.99, 0.1)
    print(returns)


def Q4_b(trials: int, num_episodes: int):
    env.register_env()
    R_1 = []
    R_2 = []
    R_3 = []

    for t in trange(trials, desc="Trials"):
        r_1 = algorithms.on_policy_mc_control_epsilon_soft(gym.make('FourRooms-v0'), num_episodes, 0.99, 0)
        R_1.append(r_1)

        r_2 = algorithms.on_policy_mc_control_epsilon_soft(gym.make('FourRooms-v0'), num_episodes, 0.99, 0.1)
        R_2.append(r_2)

        r_3 = algorithms.on_policy_mc_control_epsilon_soft(gym.make('FourRooms-v0'), num_episodes, 0.99, 0.01)
        R_3.append(r_3)

    ave_R_1 = np.average(R_1, 0)
    ave_R_2 = np.average(R_2, 0)
    ave_R_3 = np.average(R_3, 0)

    # Ave_R = np.average([ave_R_1, ave_R_2, ave_R_3])
    # upper = [np.amax(Ave_R)] * num_episodes
    # plt.plot(upper, linestyle='--', label='upper bound')

    plt.axhline(y=0.99**20, linestyle='--', label='upper bound')


    std_R_1 = np.std(R_1, 0)
    std_R_2 = np.std(R_2, 0)
    std_R_3 = np.std(R_3, 0)

    plt.plot(ave_R_1, label='ε = 0')
    plt.plot(ave_R_2, label='ε = 0.1')
    plt.plot(ave_R_3, label='ε = 0.01')

    x = np.arange(num_episodes)

    l_1 = (ave_R_1 - 1.96 * (std_R_1/np.sqrt(trials))).flatten()
    h_1 = (ave_R_1 + 1.96 * (std_R_1/np.sqrt(trials))).flatten()

    l_2 = (ave_R_2 - 1.96 * (std_R_2/np.sqrt(trials))).flatten()
    h_2 = (ave_R_2 + 1.96 * (std_R_2/np.sqrt(trials))).flatten()

    l_3 = (ave_R_3 - 1.96 * (std_R_3/np.sqrt(trials))).flatten()
    h_3 = (ave_R_3 + 1.96 * (std_R_3/np.sqrt(trials))).flatten()

    plt.fill_between(x, l_1, h_1, alpha = 0.1 )
    plt.fill_between(x, l_2, h_2, alpha = 0.1 )
    plt.fill_between(x, l_3, h_3, alpha = 0.1 )

    plt.xlabel('Num of episodes')
    plt.ylabel('Episode’s discounted return')
    plt.legend()
    plt.show()


def main():
    # Q3_a()
    # Q3_b()
    # Q4_a()
    Q4_b(10, 1000)


if __name__ == "__main__":
    main()
