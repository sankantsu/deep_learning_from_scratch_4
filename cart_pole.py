from collections import deque
import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


type State = list[float]
type Action = int
type Reward = float


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int) -> None:
        self._buffer = deque(maxlen=buffer_size)
        self._batch_size = batch_size

    def __len__(self) -> int:
        return len(self._buffer)

    def add_record(
        self,
        state: State,
        action: Action,
        reward: Reward,
        next_state: State,
        terminated: bool,
    ) -> None:
        record = (state, action, reward, next_state, terminated)
        self._buffer.append(record)

    def get_batch(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        samples = random.sample(self._buffer, self._batch_size)

        state_batch = torch.tensor([record[0] for record in samples])
        action_batch = torch.tensor([[record[1]] for record in samples])
        reward_batch = torch.tensor([[record[2]] for record in samples])
        next_state_batch = torch.tensor([record[3] for record in samples])
        is_terminated_batch = torch.tensor([[int(record[4])] for record in samples])
        return (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            is_terminated_batch,
        )


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
        self._lr = 0.0002
        self._eps = 0.1
        self._replay_buffer = ReplayBuffer(buffer_size=10000, batch_size=32)
        self._qnet = QNet()
        self._target_qnet = QNet()
        self._optimizer = torch.optim.Adam(self._qnet.parameters(), lr=self._lr)
        self._loss_f = nn.MSELoss()
        self.sync_target_qnet()

    def select_action(self, state: State) -> Action:
        state = torch.tensor(state)
        if np.random.rand() < self._eps:
            # Exploration
            return np.random.choice(self._n_actions)
        # Greedy
        qs = self._qnet(state)
        return int(torch.argmax(qs))

    def update(
        self,
        state: State,
        action: Action,
        reward: Reward,
        next_state: State,
        terminated: bool,
    ) -> None:
        self._replay_buffer.add_record(state, action, reward, next_state, terminated)
        del state, action, reward, next_state
        if len(self._replay_buffer) < self._replay_buffer._batch_size:
            return

        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            is_terminated_batch,
        ) = self._replay_buffer.get_batch()

        with torch.no_grad():
            next_qs_batch = self._qnet(next_state_batch)
            next_q_batch = torch.max(next_qs_batch, dim=1)[0].unsqueeze(1)
        non_terminated = 1 - is_terminated_batch
        target_batch = reward_batch + self._gamma * next_q_batch * non_terminated

        q_batch = self._qnet(state_batch).gather(dim=1, index=action_batch)
        loss = self._loss_f(q_batch, target_batch)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def sync_target_qnet(self) -> None:
        self._target_qnet.load_state_dict(self._qnet.state_dict())


def run_episode(env: gym.Env, agent: Agent) -> float:
    state, info = env.reset()
    episode_over = False
    total_reward = 0.0

    while not episode_over:
        action = agent.select_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        reward = float(reward)
        agent.update(state, action, reward, next_state, terminated)

        total_reward += reward
        episode_over = terminated or truncated
        state = next_state

    return total_reward


def train(agent: Agent) -> None:
    env = gym.make("CartPole-v1", render_mode=None)
    n_episodes = 1000
    sync_interval = 20

    scores = []
    for i in range(n_episodes):
        if i % 20 == 0:
            print(f"Running {i} th step...")
        total_reward = run_episode(env, agent)
        scores.append(total_reward)
        if i % sync_interval == 0:
            agent.sync_target_qnet()

    env.close()

    plt.plot(np.arange(n_episodes), scores)
    plt.show()


def test_play(agent: Agent) -> None:
    env = gym.make("CartPole-v1", render_mode="human")
    n_episodes = 10
    for i in range(n_episodes):
        total_reward = run_episode(env, agent)
        print("Score:", total_reward)
    env.close()


def main() -> None:
    agent = Agent()
    train(agent)
    test_play(agent)


if __name__ == "__main__":
    main()
