import copy
from collections import defaultdict
from collections.abc import Mapping, MutableMapping
from lib.gridworld import GridWorld


type Env = GridWorld
type State = tuple[int, int]
type Action = int
type Prob = float
type Value = float
type Strategy = Mapping[State, Mapping[Action, Prob]]
type ValueF = MutableMapping[State, Value]


def eval_step(pi: Strategy, v: ValueF, env: GridWorld, gamma: float) -> ValueF:
    for s in env.states():
        if s == env.goal_state:
            continue
        new_v = 0
        for action, p in pi[s].items():
            next_state = env.next_state(s, action)
            r = env.reward(s, action, next_state)
            new_v += p * (r + gamma * v[next_state])
        v[s] = new_v
    return v


def policy_eval(pi: Strategy, v: ValueF, env: GridWorld, gamma: float) -> ValueF:
    for _ in range(100):
        v = eval_step(pi, v, env, gamma)
    return v


def make_greedy_policy(v: ValueF, env: GridWorld, gamma: float) -> Strategy:
    new_pi = defaultdict(lambda: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0})
    for s in env.states():
        max_value = -1000000
        best_action = -1
        for action in env.action_space:
            next_state = env.next_state(s, action)
            r = env.reward(s, action, next_state)
            val = r + gamma * v[next_state]
            if val > max_value:
                max_value = val
                best_action = action
        new_pi[s][best_action] = 1.0
    return new_pi


def policy_iter(env: GridWorld, gamma: float) -> Strategy:
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    v = defaultdict(lambda: 0.0)

    while True:
        env.render_v(v, pi)
        v = policy_eval(pi, v, env, gamma)
        new_pi = make_greedy_policy(v, env, gamma)
        if new_pi == pi:
            break
        pi = new_pi
    return pi


def value_iter_step(v: ValueF, env: GridWorld, gamma: float) -> ValueF:
    v = copy.copy(v)
    for s in env.states():
        if s == env.goal_state:
            continue
        action_values = []
        for action in env.action_space:
            next_state = env.next_state(s, action)
            reward = env.reward(s, action, next_state)
            value = reward + gamma * v[next_state]
            action_values.append(value)
        v[s] = max(action_values)
    return v


def value_iter(env: GridWorld, gamma: float) -> ValueF:
    v = defaultdict(lambda: 0.0)

    threshold = 0.001
    while True:
        env.render_v(v)
        new_v = value_iter_step(v, env, gamma)

        delta = 0
        for s in env.states():
            delta = max(delta, abs(new_v[s] - v[s]))
        if delta < threshold:
            break
        v = new_v
    return v


def main() -> None:
    env = GridWorld()
    gamma = 0.9

    use_value_iter = False
    if use_value_iter:
        value_iter(env, gamma)
    else:
        policy_iter(env, gamma)


if __name__ == "__main__":
    main()
