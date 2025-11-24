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


def main() -> None:
    env = GridWorld()
    gamma = 0.9

    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    v = defaultdict(lambda: 0.0)

    v = policy_eval(pi, v, env, gamma)
    env.render_v(v, pi)


if __name__ == "__main__":
    main()
