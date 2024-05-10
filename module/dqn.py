from collections import deque
import numpy as np
import random

from keras.layers import Dense, Dropout, Input
from keras.optimizers.legacy import Adam
from keras.models import Model

import tensorflow as tf


class DQNAgent:
    def __init__(self, state_size, action_size,
                 lr=0.0001, batch_size=64, min_start=1000,
                 memory_size=10**5, epsilon_decay=0.999,
                 epsilon_min=0.1, discount_factor=0.99,
                 update_period=100) -> None:
        """
        Initialization. See belows for info.
        """
        # environment parameters
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.discount_factor = discount_factor

        # hyperparameters for the DQN
        self.lr = lr
        self.batch_size = batch_size
        self.min_start = min_start  # training starts when the data collected is more than min_start

        # memory
        self.memory_size = memory_size
        self.memory = deque(maxlen=self.memory_size)

        # create online model and target model
        self.online_model = self.create_model()
        self.target_model = self.create_model()
        self.opt = Adam(learning_rate=self.lr)

        # copy the weights of the main model to the target model
        self.update_target_model()

        self.train_step = 0
        self.update_period = update_period  # number of steps to update the target model

    def create_model(self) -> Model:
        """
        Create DNN model.
        """
        inputs = Input(shape=(self.state_size))
        x = Dense(128, activation='relu')(inputs)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.action_size, activation='linear')(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def update_target_model(self) -> None:
        """
        Copy the weights of the online model to the target models.
        This method is known as hard update, which is different from
        the soft update used in Deep Deterministic Policy Gradient.
        """
        self.target_model.set_weights(self.online_model.get_weights())

    def act(self, state, evaluate=False) -> int:
        """
        Select an action to perform based on state.

        Args:
            evaluate (bool, optional): Set to True if used for testing.
                False, as default, for training.
        """
        # exploration
        if not evaluate and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # exploitation
        else:
            q_value = self.online_model(state).numpy()[0]
            return np.argmax(q_value)

    def store_data(self, state, action, reward, next_state, done) -> None:
        """
        Store the data into the memory. Since we are using deque, the new data
        will still be added while the oldest data will be removed if the
        memory is full.
        """
        if len(self.memory) == self.min_start:
            print("Collect enough samples. Start training.")
        self.memory.append((state, action, reward, next_state, done))
        # decay the epsilon until reaching a minimum value
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def learn(self) -> None:
        """
        Train using experience replay.
        """
        # only train if the number of data points is more than min_start
        if len(self.memory) < self.min_start:
            return
        # sample a batch, convert to tensors
        minibatch = random.sample(self.memory, min(self.batch_size, len(self.memory)))
        states, actions, rewards, next_states, dones = map(tf.convert_to_tensor, zip(*minibatch))
        dones = tf.cast(dones, tf.float32)

        # Q values of next states
        q_targets_next = self.target_model(tf.squeeze(next_states))
        targets = rewards + self.discount_factor * tf.reduce_max(q_targets_next, axis=1) * (1 - dones)

        # calculate loss function and its gradient
        with tf.GradientTape() as tape:
            # Q values of current states
            q_values = self.online_model(tf.squeeze(states))
            outputs = tf.reduce_sum(q_values * tf.one_hot(actions, self.action_size), axis=1)
            # loss
            loss = tf.keras.losses.mse(targets, outputs)
            # loss = tf.reduce_mean(tf.square(targets - outputs))
            # gradients
            gradients = tape.gradient(loss, self.online_model.trainable_variables)
            # update the weights of the online model
            self.opt.apply_gradients(zip(gradients, self.online_model.trainable_variables))

        self.train_step += 1
        if self.train_step % self.update_period == 0:
            self.update_target_model()
