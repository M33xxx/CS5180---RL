import gym
import numpy as np
from tqdm import trange
from tiles3 import tiles, IHT

def e_geedy(Q, epsilon):
    if np.random.random() < epsilon:
        a = np.random.choice(np.arange(len(Q)))
    else:
        a = np.random.choice(np.where(Q == np.max(Q))[0])
    return a

def semi_grad_sarsa(env, trials, episodes, step_size, gamma, epsilon, agg):
    m = []
    for i in range(11):
        for j in range(11):
            a = [(i, j)]
            m += a

    l = list(range(len(m)))
    l = sorted(np.tile(l, agg))
    M = dict(zip(m, l))

    Tr = []
    n = env.action_space.n
    for _ in trange(trials):
        st = []
        w = np.zeros(n * len(M))

        for _ in range(episodes):
            s = env.reset()

            M_s = M[s] * n
            q_s = w[M_s: (M_s + n)]

            a = e_geedy(q_s, epsilon)
            c = 0

            while True:
                n_s, r, done, _ = env.step(a)

                M_s = M[s] * n
                M_ns = M[n_s] * n

                q_ns = w[M_ns: (M_ns + n)]
                n_a = e_geedy(q_ns, epsilon)

                grad = np.zeros(n * len(M))
                grad[M_s + a] = 1

                if done:
                    w += step_size * (r - w[M_s + a]) * grad
                    break

                w += step_size * (r + gamma * w[M_ns + n_a] - w[M_s + a]) * grad

                s = n_s
                a = n_a
                c += 1

            st.append(c)
        Tr.append(st)
    Tr = np.array(Tr)

    return Tr


def semi_grad_sarsa_extend(env, trials, episodes, step_size, gamma, epsilon):
    n = env.action_space.n + 3
    v = np.zeros(n)
    v[n-1] = 1

    Tr = []
    for _ in trange(trials):
        st = []
        w = np.zeros(n * 11*11)

        for _ in range(episodes):
            s = env.reset()
            pos = (s[0] * 11 + s[1]) * n
            q = w[pos: (pos + 4)]
            a = e_geedy(q, epsilon)
            c = 0

            while True:
                n_s, r, done, _ = env.step(a)

                pos = (s[0] * 11 + s[1]) * n
                pos_ns = (n_s[0] * 11 + n_s[1]) * n

                q_ns = w[pos_ns: (pos_ns + 4)]
                n_a = e_geedy(q_ns, epsilon)

                grad = np.tile(v, 11*11)
                grad[pos + a] = 1
                grad[pos + 4] = s[0]
                grad[pos + 5] = s[1]

                if done:
                    w += step_size * (r - w[pos + a]) * grad
                    break

                w += step_size * (r + gamma * w[pos_ns + n_a] - w[pos + a]) * grad

                s = n_s
                a = n_a
                c += 1

            st.append(c)
        Tr.append(st)
    Tr = np.array(Tr)

    return Tr

def semi_grad_sarsa_extend_more(env, trials, episodes, step_size, gamma, epsilon):
    n = env.action_space.n + 6
    v = np.zeros(n)
    v[n-1] = 1

    Tr = []
    for _ in trange(trials):
        st = []
        w = np.zeros(n * 11*11)

        for _ in range(episodes):
            s = env.reset()

            pos = (s[0] * 11 + s[1]) * n
            q = w[pos: (pos + 4)]

            a = e_geedy(q, epsilon)
            c = 0

            while True:
                n_s, r, done, _ = env.step(a)

                pos = (s[0] * 11 + s[1]) * n
                pos_ns = (n_s[0] * 11 + n_s[1]) * n
                
                q_ns = w[pos_ns: (pos_ns + 4)]
                n_a = e_geedy(q_ns, epsilon)

                s_g = env.goal_pos
                s_0 = env.start_pos
                dist_g = np.sqrt((s_g[0] - s[0]) ** 2 + (s_g[1] - s[1]) ** 2)
                dist_0 = np.sqrt((s_0[0] - s[0]) ** 2 + (s_0[1] - s[1]) ** 2)

                grad = np.tile(v, 11*11)
                grad[pos + a] = 1
                grad[pos + 4] = s[0]
                grad[pos + 5] = s[1]

                grad[pos + 6] = dist_g
                grad[pos + 7] = dist_0

                if s == s_g:
                    grad[pos + 8] = 1
                else:
                    grad[pos + 8] = 0
                
                if done:
                    w += step_size * (r - w[pos + a]) * grad
                    break

                w += step_size * (r + gamma * w[pos_ns + n_a] - w[pos + a]) * grad

                s = n_s
                a = n_a
                c += 1

            st.append(c)
        Tr.append(st)
    Tr = np.array(Tr)
    return Tr

def mountain_car_sarsa(episodes, step_size, gamma):
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 500
    
    iht = IHT(4096)
    w = np.zeros(4096)

    def Q(s, a):
        T = tiles(iht, 8, [8 * s[0] / 1.7, 8 * s[1] / 0.14], [a])
        q = 0
        for t in T:
            q += w[t]
        return q

    st = []
    for _ in range(episodes):
        s = env.reset()
        a = np.argmax([Q(s, 0), Q(s, 1), Q(s, 2)])

        c = 0
        while True:
            q = Q(s, a)
            n_s, r, done, _ = env.step(a)
            T = tiles(iht, 8, [8 * s[0] / 1.7, 8 * s[1] / 0.14], [a])

            if done:
                for t in T:
                    w[t] += step_size * (r - q)
                break

            else:
                n_a = np.argmax([Q(n_s, 0), Q(n_s, 1), Q(n_s, 2)])
                n_q = Q(n_s, n_a)

                for t in T:
                    w[t] += step_size * (r + gamma * n_q - q)

            s = n_s
            a = n_a
            c += 1
        st.append(c)

    return iht, w, st

