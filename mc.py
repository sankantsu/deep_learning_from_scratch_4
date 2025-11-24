from collections import defaultdict
import numpy as np
from lib.gridworld import GridWorld


type Action = int
type Reward = float
type State = tuple[int, int]


class Agent:
    def __init__(self, n_actions: int) -> None:
        self._n_actions = n_actions
        self._gamma = 0.9

        assert n_actions > 0
        p = 1 / n_actions
        action_probs = {i: p for i in range(n_actions)}

        self._v = defaultdict(lambda: 0)
        self._pi = defaultdict(lambda: action_probs)
        self._cnts = defaultdict(lambda: 0)
        self._memory = []

    def select_action(self, state: State) -> Action:
        actions = list(self._pi[state].keys())
        probs = list(self._pi[state].values())
        return np.random.choice(actions, p=probs)

    def update(self, state: State, action: Action, reward: Reward) -> None:
        data = (state, action, reward)
        self._memory.append(data)

    def eval(self) -> None:
        g = 0
        for data in reversed(self._memory):
            state, action, reward = data
            g = reward + self._gamma * g
            self._cnts[state] += 1
            self._v[state] += (g - self._v[state]) / self._cnts[state]

    def reset(self) -> None:
        self._memory = []


def run_episode(env: GridWorld, agent: Agent) -> None:
    env.reset()
    agent.reset()
    while True:
        state = env.agent_state
        action = agent.select_action(state)
        _, reward, done = env.step(action)
        agent.update(state, action, reward)
        if done:
            agent.eval()
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
