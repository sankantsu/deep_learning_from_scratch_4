# This example estimates value function for simple two state model.
# See section 4.1.2 of the book.


# reward r_ij: reward for transition i->j
r = [[-1, 1], [0, -1]]

# strategy s_ij: the probability to choose the action j at the state i
s = [[0.5, 0.5], [0.5, 0.5]]

# damping factor
gamma = 0.9


def update(v: list[int]) -> list[int]:
    next_v = [0, 0]
    for i in [0, 1]:
        for j in [0, 1]:
            next_v[i] += s[i][j] * (r[i][j] + gamma * v[j])
    return next_v


def main():
    v = [0, 0]  # current estimation of value function

    n_steps = 100
    for _ in range(n_steps):
        v = update(v)
    print(v)


if __name__ == "__main__":
    main()
