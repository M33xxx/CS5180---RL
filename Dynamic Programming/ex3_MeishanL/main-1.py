import algorithm


def main():
    V_a = algorithm.iterative_policy_evaluation()
    V_a = V_a[::-1]
    print ('Vs(a) = ', V_a)

    V_b, pi_s = algorithm.value_iteration()
    print ('Vs(b) = \n', V_b)
    print ('pi(s)_b = \n', pi_s)

    V_c, p = algorithm.policy_iteration()
    print ('Vs(c) = \n', V_c)
    print ('pi(s)_c = \n', p)





if __name__ == "__main__":
    main()
