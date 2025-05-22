
import numpy as np
import matplotlib
import matplotlib.pyplot as plotter
import seaborn as sns
from tqdm import trange

sns.set_style('whitegrid')

class Bandit:
    def __init__(self, k_arm=10, epsilon=0., initial=0, step_size=0.1, sample_averages=False, true_reward=0.):
        self.k = k_arm
        self.epsilon = epsilon
        self.initial = initial
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.true_reward = true_reward

        self.arm_indices = np.arange(self.k)
        self.time = 0
        self.average_reward = 0

    # get an action for this bandit
    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.arm_indices)
        
        q_best = np.max(self.q_estimation)
        return np.random.choice(np.where(self.q_estimation == q_best)[0])

    # Perform the action, and update estimates for this action.
    def perform(self, action):
        reward = np.random.randn() + self.q_true[action]
        self.time += 1
        self.action_count_arms[action] += 1
        self.average_reward += (reward - self.average_reward) / self.time

        if self.sample_averages:
            # Update estimates using Sample averages
            self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count_arms[action]
        else:
            # Update estimates using constant step size
            self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])

        return reward

    def reset(self):
        # real reward for each action
        self.q_true = np.random.randn(self.k) + self.true_reward

        # estimated q value for each action
        self.q_estimation = np.zeros(self.k) + self.initial

        # no. of times each arm is chosen.
        self.action_count_arms = np.zeros(self.k)

        self.best_action = np.argmax(self.q_true)
        self.time = 0

def execute(bandits, runs, time):
    rewards = np.zeros((len(bandits), runs, time))
    best_action_counts = np.zeros(rewards.shape)
    for i, bandit in enumerate(bandits):
        for r in trange(runs):
            bandit.reset()
            for t in range(time):
                action = bandit.get_action()
                reward = bandit.perform(action)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1
    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    return mean_best_action_counts, mean_rewards


def plot():
    epsilons = [0, 0.1, 0.2]
    graph_colors = ['green', 'blue', 'red']

    bandits = [Bandit(epsilon=epsilon, sample_averages=True) for epsilon in epsilons]
    best_action_counts, rewards = execute(runs=2000, time=1000, bandits=bandits)

    fig, axes = plotter.subplots(2, 1, figsize=(10, 20), dpi=120)
    # fig.subplots_adjust(top=0.2, bottom=0.2)

    index=0
    for epsilon, rewards in zip(epsilons, rewards):
        axes[0].plot(rewards, color=graph_colors[index], label=r'$\epsilon = %0.1f$' % (epsilon))
        index += 1

    axes[0].set_xlabel('steps', fontsize=14)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    # axes[0].spines['bottom'].set_visible(False)

    axes[0].set_xlabel('Steps', fontsize=18)
    axes[0].set_ylabel('Average reward', fontsize=18)
    axes[0].grid(True)
    axes[0].text(200, 1.4, r'$\epsilon = 0.1$', color='blue', fontsize=16, fontweight='bold')
    axes[0].text(1000, 1.15, r'$\epsilon = 0.2$', color='red', fontsize=16, fontweight='bold')
    axes[0].text(400, 0.92, r'$\epsilon = 0 (greedy)$', color='green', fontsize=16, fontweight='bold')

    index=0
    for epsilon, counts in zip(epsilons, best_action_counts):
        axes[1].plot(counts, color=graph_colors[index], label=r'$\epsilon = %0.1f$' % (epsilon))
        index += 1

    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    axes[1].set_xlabel('Steps', fontsize=18)
    axes[1].set_ylabel('% Optimal action', fontsize=18)
    axes[1].grid(True)
    # axes[1].legend(loc='upper right')

    axes[1].text(340, 0.8, r'$\epsilon = 0.1$', color='blue', fontsize=16, fontweight='bold')
    axes[1].text(1000, 0.7, r'$\epsilon = 0.2$', color='red', fontsize=16, fontweight='bold')
    axes[1].text(400, 0.32, r'$\epsilon = 0 (greedy)$', color='green', fontsize=16, fontweight='bold')

    # plotter.tight_layout()
    fig.suptitle('Multi-armed Bandit with 10 arms Plot', fontsize=22, fontweight='bold')
    # plotter.show()
    plotter.savefig('../diagrams/ar_oa_plot.png')


if __name__ == '__main__':
    plot()