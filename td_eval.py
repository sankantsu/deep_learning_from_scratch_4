from collections import defaultdict
from collections.abc import MutableMapping
import numpy as np
from lib.gridworld import GridWorld


type Prob = float
type Action = int
type Score = float
type Reward = Score
type State = tuple[int, int]
type QKey = tuple[State, Action]
type Q = MutableMapping[QKey, Score]
type Record = tuple[State, Action, Reward]


class Agent:
    def __init__(self, n_actions: int) -> None:
        assert n_actions > 0
        p = 1 / n_actions
        default_action_probs = {i: p for i in range(n_actions)}

        self._gamma: float = 0.9
        self._alpha = 0.01
        self._v: MutableMapping[State, Score] = defaultdict(lambda: 0.0)
        self._pi: MutableMapping[State, dict[Action, Prob]] = defaultdict(
            lambda: default_action_probs
        )

    def select_action(self, state: State) -> Action:
        actions = list(self._pi[state].keys())
        probs = list(self._pi[state].values())
        return np.random.choice(actions, p=probs)

    def eval(self, state: State, next_state: State, reward: Reward, done: bool) -> None:
        self._v[state] += self._alpha * (
            reward + self._gamma * self._v[next_state] - self._v[state]
        )


def run_episode(env: GridWorld, agent: Agent) -> None:
    env.reset()
    while True:
        state = env.agent_state
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.eval(state, next_state, reward, done)
        if done:
            return


def main():
    env = GridWorld()
    agent = Agent(n_actions=4)

    n_episode = 1000
    for _ in range(n_episode):
        run_episode(env, agent)
    env.render_v(agent._v)


if __name__ == "__main__":
    main()
