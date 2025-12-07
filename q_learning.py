from collections import defaultdict
from collections.abc import MutableMapping
from typing import cast
import numpy as np
from lib.gridworld import GridWorld


type Prob = float
type Action = int
type Score = float
type Reward = Score
type State = tuple[int, int]
type QKey = tuple[State, Action]
type Record = tuple[State, Action, Reward]


def make_eps_greedy_probs(
    q: MutableMapping[QKey, Score], state: State, n_actions: int, eps: float
) -> dict[Action, Prob]:
    qs = [q[(state, action)] for action in range(n_actions)]
    best_action = np.argmax(qs)
    p = eps / n_actions
    action_probs = {i: p for i in range(n_actions)}
    action_probs[best_action] += 1 - eps
    return action_probs


class Agent:
    def __init__(self, n_actions: int) -> None:
        assert n_actions > 0
        p = 1 / n_actions

        self._n_actions = n_actions
        self._gamma: float = 0.9
        self._alpha = 0.1
        self._eps = 0.2
        self._Q: MutableMapping[QKey, Score] = defaultdict(lambda: 0)

    def select_action(self, state: State) -> Action:
        if np.random.rand() < self._eps:
            # Exploration
            return np.random.choice(self._n_actions)
        # Greedy
        qs = [self._Q[state, action] for action in range(self._n_actions)]
        return cast(Action, np.argmax(qs))

    def update(
        self, state: State, action: Action, reward: Reward, next_state: State
    ) -> None:
        next_q = max(self._Q[next_state, action] for action in range(self._n_actions))
        target = reward + self._gamma * next_q
        self._Q[state, action] += self._alpha * (target - self._Q[state, action])

    def reset(self) -> None:
        pass


def run_episode(env: GridWorld, agent: Agent) -> None:
    env.reset()
    agent.reset()
    while True:
        state = env.agent_state
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        if done:
            return


def main():
    env = GridWorld()
    agent = Agent(n_actions=4)

    n_episode = 10000
    for _ in range(n_episode):
        run_episode(env, agent)
    env.render_q(agent._Q)


if __name__ == "__main__":
    main()
