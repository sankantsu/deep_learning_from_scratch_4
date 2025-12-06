# SARSA
# Q(s_t, a_t) <- (1 - \alpha) * Q(s_t, a_t)
#                + \alpha * (R_t + \gamma * Q(s_{t+1}, a_{t+1}))
#
# My interpretation:
# Q value should be updated with reward sample (R_t) + estimated value for next
# state (\gamma * v(s_{t+1})). It is hard to estimate the state value without
# environment model. Even if it is possible, it is computationally expensive.
# Therefore, we will approximate the state value estimation with the Q value
# of (state, action) sample in the trajectory.

from collections import defaultdict, deque
from collections.abc import MutableMapping
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
        default_action_probs = {i: p for i in range(n_actions)}

        self._n_actions = n_actions
        self._gamma: float = 0.9
        self._alpha = 0.1
        self._eps = 0.2
        self._Q: MutableMapping[QKey, Score] = defaultdict(lambda: 0)
        self._pi: MutableMapping[State, dict[Action, Prob]] = defaultdict(
            lambda: default_action_probs
        )
        self._memory: deque[tuple[State, Action, Reward]] = deque(maxlen=2)

    def select_action(self, state: State) -> Action:
        actions = list(self._pi[state].keys())
        probs = list(self._pi[state].values())
        return np.random.choice(actions, p=probs)

    def update(self, state: State, action: Action, reward: Reward, done: bool) -> None:
        self._memory.append((state, action, reward))
        del state, action, reward
        if len(self._memory) < 2:
            return
        s1, a1, r = self._memory[0]
        s2, a2, _ = self._memory[1]

        target = r + self._gamma * self._Q[s2, a2]
        self._Q[s1, a1] += self._alpha * (target - self._Q[s1, a1])

        self._pi[s1] = make_eps_greedy_probs(self._Q, s1, self._n_actions, self._eps)

        # Update the Q for the last action
        if done:
            s, a, r = self._memory[1]
            target = r
            self._Q[s, a] += self._alpha * (target - self._Q[s, a])
            self._pi[s] = make_eps_greedy_probs(self._Q, s, self._n_actions, self._eps)

    def reset(self) -> None:
        self._memory.clear()


def run_episode(env: GridWorld, agent: Agent) -> None:
    env.reset()
    agent.reset()
    while True:
        state = env.agent_state
        action = agent.select_action(state)
        _, reward, done = env.step(action)
        agent.update(state, action, reward, done)
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
