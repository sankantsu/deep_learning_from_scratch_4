import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


type State = list[float]
type Action = int
type Reward = float


class QNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        observation_dim = 4
        hidden_size = 128
        action_size = 2
        self.l1 = nn.Linear(observation_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.relu(self.l1(x))
        x = nn.functional.relu(self.l2(x))
        return self.l3(x)


class Agent:
    def __init__(self) -> None:
        self._n_actions = 2
        self._gamma = 0.98
        self._lr = 0.001
        self._eps = 0.1
        self._qnet = QNet()
        self._optimizer = torch.optim.Adam(self._qnet.parameters(), lr=self._lr)
        self._loss_f = nn.MSELoss()

    def select_action(self, state: State) -> Action:
        state = torch.tensor(state)
        if np.random.rand() < self._eps:
            # Exploration
            return np.random.choice(self._n_actions)
        # Greedy
        qs = self._qnet(state)
        return int(torch.argmax(qs))

    def update(self, state: State, action: Action, reward: Reward, next_state: State) -> None:
        state = torch.tensor(state)
        next_state = torch.tensor(next_state)

        next_qs = self._qnet(next_state).detach()
        next_q = max(next_qs)
        target = reward + self._gamma * next_q
        
        q = self._qnet(state)[action]
        loss = self._loss_f(q, target)
        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()


def run_episode(env: gym.Env, agent: Agent):
    state, info = env.reset()
    episode_over = False
    total_reward = 0.

    while not episode_over:
        action = agent.select_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        reward = float(reward)

        total_reward += reward
        episode_over = terminated or truncated
        agent.update(state, action, reward, next_state)
        state = next_state

    return total_reward


def main():
    env = gym.make("CartPole-v1", render_mode="human")
    agent = Agent()
    n_episodes = 1000

    scores = []
    for i in range(n_episodes):
        if i % 1000 == 0:
            print(f"Running {i} th step...")
        total_reward = run_episode(env, agent)
        scores.append(total_reward)

    env.close()

    plt.plot(np.arange(n_episodes), scores)
    plt.show()


if __name__ == "__main__":
    main()
