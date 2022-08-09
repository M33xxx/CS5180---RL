import numpy as np
import env

def iterative_policy_evaluation():
    delta = 1
    theta = 10 ** (-3)

    w = env.Gridworld5x5()
    V_s = np.zeros((5, 5), dtype=float)

    while delta > theta:
        delta = 0

        for s in w.state_space:
            v = V_s[s]
            temp = 0

            for a in range(w.action_space):
                prob = 1/4
                temp += prob * w.expected_return(V_s, s, env.Action(a), 0.9)

            V_s[s] = temp
            delta = max(delta, np.abs(v - V_s[s]))

    return V_s

def num_to_string(p):
    p_print = np.zeros((5, 5), dtype=object)

    for i in range(5):
        for j in range(5):
            num = p[i][j]
            if num == 0:
                p_print[i][j] = 'up'
            elif num == 1:
                p_print[i][j] = 'left'
            elif num == 2:
                p_print[i][j] = 'down'
            elif num == 3:
                p_print[i][j] = 'right'

    return p_print

def value_iteration():
    delta = 1
    theta = 10 ** (-3)

    w = env.Gridworld5x5()
    V_s = np.zeros((5, 5), dtype=float)
    pi_s = np.zeros((5, 5), dtype=float)

    while delta > theta:
        delta = 0

        for s in w.state_space:
            v = V_s[s]
            v_a = np.zeros(w.action_space)

            for a in range(w.action_space):
                v_a[a] = w.expected_return(V_s, s, env.Action(a), 0.9)

            V_s[s] = np.max(v_a)
            pi_s[s] = np.argmax(v_a)
            delta = max(delta, np.abs(v - V_s[s]))

    p_print = num_to_string(pi_s)

    return V_s, p_print

def policy_iteration():
    theta = 10 ** (-3)
    w = env.Gridworld5x5()
    V_s = np.zeros((5, 5), dtype=float)

    def policy_evaluation(p):
        delta = 1
        while delta > theta:
            delta = 0

            for s in w.state_space:
                v = V_s[s]
                a = p[s]
                V_s[s] = w.expected_return(V_s, s, a, 0.9)
                delta = max(delta, np.abs(v - V_s[s]))

        return V_s

    "Policy improvement"
    s = True
    p = np.zeros((5, 5), dtype=float)
    o_a = p.copy()
    V_s = policy_evaluation(p)

    for s in w.state_space:
        v_a = np.zeros(w.action_space)

        for a in range(w.action_space):
            v_a[a] = w.expected_return(V_s, s, env.Action(a), 0.9)

        p[s] = np.argmax(v_a)
        V_s[s] = np.max(v_a)

        if o_a[s] != p[s]:
            s = False

    p_print = num_to_string(p)

    if s:
            return V_s, p_print


