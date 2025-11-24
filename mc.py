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


def make_eps_greedy_probs(
    q: Q, state: State, n_actions: int, eps: float
) -> dict[Action, Prob]:
    qs = [q[(state, action)] for action in range(n_actions)]
    best_action = np.argmax(qs)
    p = eps / n_actions
    action_probs = {i: p for i in range(n_actions)}
    action_probs[best_action] += 1 - eps
    return action_probs


class Agent:
    def __init__(self, n_actions: int) -> None:
        self._n_actions: int = n_actions
        self._gamma: float = 0.9
        self._eps: float = 0.1  # exploration rate
        self._alpha: float = 0.1  # weight for new observation

        assert n_actions > 0
        p = 1 / n_actions
        action_probs = {i: p for i in range(n_actions)}

        self._Q: Q = defaultdict(lambda: 0)
        self._pi: MutableMapping[State, dict[Action, Prob]] = defaultdict(
            lambda: action_probs
        )
        self._cnts: MutableMapping[State, int] = defaultdict(lambda: 0)
        self._memory: list[Record] = []

    def select_action(self, state: State) -> Action:
        actions = list(self._pi[state].keys())
        probs = list(self._pi[state].values())
        return np.random.choice(actions, p=probs)

    def add_record(self, state: State, action: Action, reward: Reward) -> None:
        data = (state, action, reward)
        self._memory.append(data)

    def update_policy(self) -> None:
        g = 0
        for data in reversed(self._memory):
            state, action, reward = data
            g = reward + self._gamma * g
            self._cnts[state] += 1
            key = (state, action)
            self._Q[key] += (g - self._Q[key]) * self._alpha
            self._pi[state] = make_eps_greedy_probs(
                self._Q, state, self._n_actions, self._eps
            )

    def reset(self) -> None:
        self._memory = []


def run_episode(env: GridWorld, agent: Agent) -> None:
    env.reset()
    agent.reset()
    while True:
        state = env.agent_state
        action = agent.select_action(state)
        _, reward, done = env.step(action)
        agent.add_record(state, action, reward)
        if done:
            agent.update_policy()
            return


def main():
    env = GridWorld()
    agent = Agent(n_actions=4)

    n_episode = 1000
    for _ in range(n_episode):
        run_episode(env, agent)
    env.render_q(agent._Q)


if __name__ == "__main__":
    main()
