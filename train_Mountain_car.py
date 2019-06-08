import gym
import numpy as np


class Mountain:

    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        self.episodes = 2000
        self.learning_rate = 0.01
        self.discount = 0.95
        self.epsilon = 0.5
        self.epsilon_decay_end_range = 2
        self.end_epsilon_decay = self.episodes//self.epsilon_decay_end_range
        self.epsilon_decay_rate = self.epsilon/(self.end_epsilon_decay - 1)
        self.discrete_window_size = 20
        self.discrete_observation_size = [self.discrete_window_size] * len(self.env.observation_space.high)
        self.discrete_observation_window_size = (self.env.observation_space.high - self.env.observation_space.low)/self.discrete_observation_size
        self.q_table = np.random.uniform(low=-2, high=0, size=(self.discrete_observation_size + [self.env.action_space.n]))
        self.episodes_reward_tracker = []

    def set_parameters(self):
        pass

    def get_discrete_state(self, state):
        discrete_state = (state - self.env.observation_space.low) / self.discrete_observation_window_size
        return tuple(discrete_state.astype(np.int))

    def train(self, render_every=500, episodes=25000):
        self.episodes = episodes
        for episode in range(self.episodes):
            episode_reward = 0
            if episode % render_every == 0:
                print(episode)
                render = True
            else:
                render = False
            discrete_state = self.get_discrete_state(self.env.reset())
            done = False
            while not done:
                if np.random.random() > self.epsilon:
                    action = np.argmax(self.q_table[discrete_state])
                else:
                    action = np.random.randint(0, self.env.action_space.n)
                new_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                new_discrete_state = self.get_discrete_state(new_state)

                if render:
                    self.env.render()
                if not done:
                    max_future_q = np.max(self.q_table[new_discrete_state])
                    current_q = self.q_table[discrete_state + (action,)]
                    learned_value = (reward + self.discount * max_future_q)
                    # in Q learning formula last half is called learned value
                    new_q = (1 - self.learning_rate) * current_q + self.learning_rate * learned_value

                    self.q_table[discrete_state + (action,)] = new_q

                elif new_state[0] >= 0.5:
                    print(f"Run successfully completed on it on episode {episode}")
                    self.q_table[discrete_state + (action,)] = 0

                discrete_state = new_discrete_state

            if self.end_epsilon_decay > episode >= 1:
                self.epsilon -= self.epsilon_decay_rate

            self.episodes_reward_tracker.append(episode_reward)
        self.env.close()


if __name__ == '__main__':
    trainer = Mountain()
    trainer.train()
