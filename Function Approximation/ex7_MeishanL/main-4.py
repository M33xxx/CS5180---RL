import env
import Algorithms
import gym
import numpy as np
from tqdm import trange
from tiles3 import tiles
import matplotlib.pyplot as plt

def Q3b():
    env.register_env()
    Env = gym.make('FourRooms-v0')
    Tr = Algorithms.semi_grad_sarsa(env=Env, trials=100, episodes=100, step_size=0.1, gamma=0.99, epsilon=0.1, agg=1)
    y = np.average(Tr, axis=0)
    x = list(range(Tr.shape[1]))

    y_std = np.std(Tr, 0)
    l = y - 1.96 * y_std / np.sqrt(Tr.shape[0])
    h = y + 1.96 * y_std / np.sqrt(Tr.shape[0])

    plt.figure()
    plt.plot(x, y, label='sarsas')
    plt.fill_between(x, l, h, alpha=0.2)

    plt.xlabel("episodes")
    plt.ylabel("step per episodes")
    plt.title("Semi-grad one step SARSA ")
    plt.legend()
    plt.show()

def Q3c():
    env.register_env()
    Env = gym.make('FourRooms-v0')

    Tr1 = Algorithms.semi_grad_sarsa(env=Env, trials=100, episodes=100, step_size=0.1, gamma=0.99, epsilon=0.1, agg=1)
    Tr2 = Algorithms.semi_grad_sarsa(env=Env, trials=100, episodes=100, step_size=0.1, gamma=0.99, epsilon=0.1, agg=2)
    Tr3 = Algorithms.semi_grad_sarsa(env=Env, trials=100, episodes=100, step_size=0.1, gamma=0.99, epsilon=0.1, agg=3)
    Tr4 = Algorithms.semi_grad_sarsa(env=Env, trials=100, episodes=100, step_size=0.1, gamma=0.99, epsilon=0.1, agg=4)

    x1 = list(range(Tr1.shape[1]))
    x2 = list(range(Tr2.shape[1]))
    x3 = list(range(Tr3.shape[1]))
    x4 = list(range(Tr4.shape[1]))

    y1 = np.average(Tr1, axis=0)
    y2 = np.average(Tr2, axis=0)
    y3 = np.average(Tr3, axis=0)
    y4 = np.average(Tr4, axis=0)

    y1_std = np.std(Tr1, 0)
    y2_std = np.std(Tr2, 0)
    y3_std = np.std(Tr3, 0)
    y4_std = np.std(Tr4, 0)

    l1 = y1 - 1.96 * y1_std / np.sqrt(Tr1.shape[0])
    l2 = y2 - 1.96 * y2_std / np.sqrt(Tr2.shape[0])
    l3 = y3 - 1.96 * y3_std / np.sqrt(Tr3.shape[0])
    l4 = y4 - 1.96 * y4_std / np.sqrt(Tr4.shape[0])

    h1 = y1 + 1.96 * y1_std / np.sqrt(Tr1.shape[0])
    h2 = y2 + 1.96 * y2_std / np.sqrt(Tr2.shape[0])
    h3 = y3 + 1.96 * y3_std / np.sqrt(Tr3.shape[0])
    h4 = y4 + 1.96 * y4_std / np.sqrt(Tr4.shape[0])

    plt.figure(1)
    plt.plot(x1, y1, label='sarsas-1')
    plt.fill_between(x1, l1, h1, alpha=0.2)
    plt.xlabel("episodes")
    plt.ylabel("step per episodes")
    plt.title("Semi-grad one step SARSA ")
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.plot(x2, y2, label='sarsas-2')
    plt.fill_between(x2, l2, h2, alpha=0.2)
    plt.xlabel("episodes")
    plt.ylabel("step per episodes")
    plt.title("Semi-grad one step SARSA ")
    plt.legend()
    plt.show()

    plt.figure(3)
    plt.plot(x3, y3, label='sarsas-3')
    plt.fill_between(x3, l3, h3, alpha=0.2)
    plt.xlabel("episodes")
    plt.ylabel("step per episodes")
    plt.title("Semi-grad one step SARSA ")
    plt.legend()
    plt.show()

    plt.figure(4)
    plt.plot(x4, y4, label='sarsas-4')
    plt.fill_between(x4, l4, h4, alpha=0.2)
    plt.xlabel("episodes")
    plt.ylabel("step per episodes")
    plt.title("Semi-grad one step SARSA ")
    plt.legend()
    plt.show()

def Q3d():
    env.register_env()

    Tr = Algorithms.semi_grad_sarsa_extend(env=gym.make('FourRooms-v0'), trials=100, episodes=100, step_size=0.1, gamma=0.99, epsilon=0.1)
    y = np.average(Tr, axis=0)
    x = list(range(Tr.shape[1]))
    y_std = np.std(Tr, 0)
    l = y - 1.96 * y_std / np.sqrt(Tr.shape[0])
    h = y + 1.96 * y_std / np.sqrt(Tr.shape[0])

    plt.figure(2)
    plt.plot(x, y, label='extended features sarsa')
    plt.fill_between(x, l, h, alpha=0.2)

    plt.xlabel("episodes")
    plt.ylabel("step per episodes")
    plt.title("Semi-grad SARSA extended features")
    plt.legend()
    plt.show()

def Q3e():
    env.register_env()

    Tr = Algorithms.semi_grad_sarsa_extend_more(env=gym.make('FourRooms-v0'), trials=100, episodes=100, step_size=0.1, gamma=0.99, epsilon=0.1)
    y = np.average(Tr, axis=0)
    x = list(range(Tr.shape[1]))
    y_std = np.std(Tr, 0)
    l = y - 1.96 * y_std / np.sqrt(Tr.shape[0])
    h = y + 1.96 * y_std / np.sqrt(Tr.shape[0])

    plt.figure(2)
    plt.plot(x, y, label='extended features sarsa')
    plt.fill_between(x, l, h, alpha=0.2)

    plt.xlabel("episodes")
    plt.ylabel("step per episodes")
    plt.title("Semi-grad SARSA extended more features")
    plt.legend()
    plt.show()

def Q4(trails, episodes):
    Tr1 = []
    Tr2 = []
    Tr3 = []

    for _ in trange(trails):
        _, _, st1 = Algorithms.mountain_car_sarsa(episodes, step_size=0.1 / 8, gamma=1)
        _, _, st2 = Algorithms.mountain_car_sarsa(episodes, step_size=0.2 / 8, gamma=1)
        _, _, st3 = Algorithms.mountain_car_sarsa(episodes, step_size=0.5 / 8, gamma=1)

        Tr1.append(st1)
        Tr2.append(st2)
        Tr3.append(st3)

    Tr1_ave = np.average(Tr1, 0)
    Tr2_ave = np.average(Tr2, 0)
    Tr3_ave = np.average(Tr3, 0)

    plt.figure()
    plt.plot(Tr1_ave, label='step_size = 0.1/8')
    plt.plot(Tr2_ave, label='step_size = 0.2/8')
    plt.plot(Tr3_ave, label='step_size = 0.5/8')

    plt.title('Mountain car learning curves')
    plt.xlabel('episodes')
    plt.ylabel('steps per episode')
    plt.legend()
    plt.show()

def Q4_3d(episodes):
    iht, w, _ = Algorithms.mountain_car_sarsa(episodes, step_size=0.1 / 8, gamma=1)
    z = np.zeros([40, 40])

    for i in range(40):
        for j in range(40):
            for a in [0, 1, 2]:
                s = [-1.2 + (1.7/40) * i, -0.07 + (0.14/40) * j]
                T = tiles(iht, 8, [8 * s[0] / 1.7, 8 * s[1] / 0.14], [a])

                q = 0
                for t in T:
                    q += w[t]
                Q = -q
            z[i, j] = Q

    x = np.arange(z.shape[0])
    y = np.arange(z.shape[1])
    x, y= np.meshgrid(x, y)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='summer', linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=7)

    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('Mountain car episode = 9000')
    plt.show()

def main():
    # Q3b()
    # Q3c()
    # Q3d()
    # Q3e()
    # Q4(trails=100, episodes=500)
    Q4_3d(episodes=9000)

if __name__ == "__main__":
    main()