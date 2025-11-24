import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, n_arms: int) -> None:
        self._rates = np.random.rand(n_arms)

    def play(self, i: int) -> int:
        rate = self._rates[i]
        r = np.random.rand()
        return 1 if rate > r else 0


class Agent:
    def __init__(self, n_actions: int, eps: float) -> None:
        self._Qs = np.zeros(n_actions)
        self._ns = np.zeros(n_actions)
        self._eps = eps

    def update(self, action: int, reword: float) -> None:
        self._ns[action] += 1
        self._Qs[action] += (reword - self._Qs[action]) / self._ns[action]

    def select_action(self) -> int:
        r = np.random.rand()
        if r < self._eps:
            # Exploration
            n_actions = len(self._Qs)
            return np.random.randint(0, n_actions)
        # Exploitation
        return np.argmax(self._Qs)


def play(bandit: Bandit, agent: Agent, n_steps: int) -> list[int]:
    rewards = []
    for i in range(n_steps):
        action = agent.select_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        rewards.append(reward)
    return rewards


def main() -> None:
    n_arms = 10
    eps = 0.1

    bandit = Bandit(n_arms)
    agent = Agent(n_arms, eps)
    print("Rates:", bandit._rates)

    n_steps = 1000
    rewards = play(bandit, agent, n_steps)

    rewards = np.array(rewards)
    total_rewards = np.cumsum(rewards)

    score = total_rewards[-1]
    print("Score:", score)

    idx = np.arange(0, n_steps) + 1

    # Total rewards
    plt.plot(idx, total_rewards)
    plt.xlabel("Steps")
    plt.ylabel("Total rewards")
    plt.show()

    # Win rate
    rates = total_rewards / idx
    plt.plot(idx, rates)
    plt.xlabel("Steps")
    plt.ylabel("Win rate")
    plt.show()


if __name__ == "__main__":
    main()
