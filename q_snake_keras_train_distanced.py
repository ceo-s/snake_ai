from snake_v0_ai import Game, RenderMode

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense, ReLU
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt

import random
import pickle
from collections import deque
import gc
import sys

gc.collect()

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 1e-3
GAMMA = 0.9
GAME_FIELD_SIZE = 24
STATE_VECTOR_SIZE = 18
GAMES_EPSILON = int(sys.argv[1]) if len(sys.argv) >= 2 else 80
WEIGHTS_DIRECTORY = "value_based/q/distanced"


class Trainer:

    def __init__(self, model: Model, lr, gamma) -> None:
        self.lr = lr
        self.gamma = gamma
        optimizer = Adam(lr)
        model.compile(optimizer, "mse")
        self.model = model

    def train_step(self, states, actions, rewards, next_states, dones):
        pred = self.model(states).numpy()
        target = np.copy(pred)

        for i, done in enumerate(dones):
            q_new = rewards[i]
            if not done:
                q_new = rewards[i] + self.gamma * \
                    np.max(self.model(next_states[i][None, ...]).numpy())
                target[i][actions[i] + 1] = q_new

        self.model.fit(states, target, verbose=False)


class Agent:

    def __init__(self, model: Model, trainer: Trainer):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = model
        self.trainer = trainer
        self.stats = {
            "scores": [],
            "rewards": [],
        }

    def __del__(self):
        self.backup()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(
            state[None, ...],
            np.array([action]),
            np.array([reward]),
            np.array([next_state]),
            np.array([done]),
        )

    def train_long_memory(self):
        batch = random.sample(self.memory, BATCH_SIZE) \
            if len(self.memory) > BATCH_SIZE \
            else self.memory

        params = [np.array(param) for param in zip(*batch)]

        self.trainer.train_step(*params)

    def generate_action(self, state):
        self.epsilon = GAMES_EPSILON - self.n_games

        if random.randint(0, 200) < self.epsilon:
            action = random.choice([-1, 0, 1])

        else:
            action = self.model(state[None, ...]).numpy()
            action = np.argmax(action) - 1

        return action

    def plot_statistics(self):
        fig, (lax, rax) = plt.subplots(1, 2, figsize=(15, 5))
        lax.plot(self.stats["scores"])
        lax.set_title(f"Scores")
        rax.plot(self.stats["rewards"])
        rax.set_title(f"Rewards")
        fig.suptitle(f"{self.n_games} games")
        plt.show()

    def backup(self):
        self.model.save_weights(
            f"{WEIGHTS_DIRECTORY}/{self.n_games}.keras")


model = Sequential([
    Input((STATE_VECTOR_SIZE, )),
    Flatten(),
    Dense(256, activation="relu"),
    Dense(36, activation="relu"),
    Dense(3)
])

trainer = Trainer(model, LR, GAMMA)
agent = Agent(model, trainer)

env = Game(GAME_FIELD_SIZE, mode=RenderMode.DISTANCED_VECTOR, verbose=True)


def start_epoch(agent: Agent):
    steps_without_apple = 0
    done = False
    old_state, info = env.reset()
    total_reward = 0
    print(old_state)
    while not done:
        steps_without_apple += 1
        action = agent.generate_action(old_state)
        new_state, reward, done, _ = env.step(action)

        if reward == 10:
            steps_without_apple = 0

        # elif reward == 0.01:
            # reward = -0.01

        if steps_without_apple > GAME_FIELD_SIZE**2:
            reward = -10
            break

        print(reward)
        agent.train_short_memory(old_state, action, reward, new_state, done)
        agent.remember(old_state, action, reward, new_state, done)
        old_state = new_state

        total_reward += reward

    return env.score - 3, total_reward


def train(agent: Agent, epochs: int, backup_frequency: int = 100, verbose: bool = True):

    for i in range(epochs):
        score, reward = start_epoch(agent)
        agent.n_games += 1
        agent.stats["scores"].append(score)
        agent.stats["rewards"].append(reward)
        agent.train_long_memory()
        agent.plot_statistics()

        if (i + 1) % backup_frequency == 0:
            agent.backup()


if __name__ == "__main__":
    train(agent, 1, 50)
