import numpy as np
import random

import tensorflow as tf
from keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.initializers import RandomUniform

from collections import deque

# Original paper: CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING: https://arxiv.org/pdf/1509.02971


class DDPGAgent():
    def __init__(self, state_size, action_size,
                 lr_actor=0.001, lr_critic=0.001,
                 gamma=0.95, batch_size=64,
                 memory_size=10**6, min_start=10000,
                 min_action=-1, max_action=1, noise=0.1,
                 replace_step=500) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory_size = memory_size

        self.memory = deque(maxlen=self.memory_size)
        self.min_start = min_start

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self.min_action = min_action
        self.max_action = max_action

        self.actor_main = self.build_actor_network()
        self.actor_target = self.build_actor_network()

        self.critic_main = self.build_critic_network()
        self.critic_target = self.build_critic_network()

        self.update_target(tau=1)

        self.opt_actor = tf.keras.optimizers.legacy.Adam(learning_rate=self.lr_actor)
        self.opt_critic = tf.keras.optimizers.legacy.Adam(learning_rate=self.lr_critic)

        self.train_step = 0
        self.replace_step = replace_step

        self.noise = noise

    def build_actor_network(self):
        """
        The actor network
        """
        last_init = RandomUniform(minval=-0.003, maxval=0.003)

        inputs = Input(shape=(self.state_size,))
        out = Dense(256, activation="relu")(inputs)
        out = Dense(256, activation="relu")(out)
        outputs = Dense(self.action_size, activation="tanh", kernel_initializer=last_init)(out)
        outputs = outputs * self.max_action

        model = Model(inputs, outputs)

        return model

    def build_critic_network(self):
        """
        The critic network used for estimating the value function
        The input is [state, action], output is the Q(s, a)
        """
        state_input = Input(shape=(self.state_size,))
        state_out = Dense(32, activation="relu")(state_input)

        # Action as input
        action_input = Input(shape=(self.action_size,))
        action_out = Dense(32, activation="relu")(action_input)

        # Both are passed through separate layer before concatenating
        concat = Concatenate()([state_out, action_out])

        out = Dense(256, activation="relu")(concat)
        out = Dense(256, activation="relu")(out)
        outputs = Dense(1)(out)

        # Outputs single value for give state-action
        model = Model([state_input, action_input], outputs)

        return model

    def act(self, state, evaluate=False):
        """
        Return the action value based on the input state

        evaluate (bool, optional): False for training, True for testing.
        """
        actions = self.actor_main(state)
        if not evaluate:
            # we add noise for exploration
            actions += np.random.normal(0, self.noise, (1, self.action_size))
        # we clip it since it might be out of range after adding noise
        actions = np.clip(actions, self.min_action, self.max_action)

        return actions[0]

    def store_data(self, state, action, reward, next_state, done):
        """
        Store data into the memory.
        """
        if len(self.memory) == self.min_start:
            print("Collect enough samples, training starting")
        # Append the new data to the memory
        self.memory.append([state, action, reward, next_state, done])

    def update_target(self, tau=0.005):
        """
        Updae the target model using soft update.
        """
        for (weight, target) in zip(self.actor_main.weights, self.actor_target.weights):
            # update the target values
            target.assign(weight * tau + target * (1 - tau))

        for (weight, target) in zip(self.critic_main.weights, self.critic_target.weights):
            # update the target values
            target.assign(weight * tau + target * (1 - tau))

    def learn(self):
        """
        Training using the samples from memory.
        """
        if len(self.memory) < self.min_start:
            return
        # sample a minibatch from the memory
        minibatch = random.sample(self.memory, min(self.memory_size, self.batch_size))
        states, actions, rewards, next_states, dones = [tf.convert_to_tensor(x, dtype=tf.float32) for x in zip(*minibatch)]

        states = tf.squeeze(states)
        rewards = tf.reshape(rewards, shape=(-1, 1))
        next_states = tf.squeeze(next_states)

        # Critic loss
        with tf.GradientTape() as tape1:
            actions_next_states = self.actor_target(next_states)
            Q_value_next_states = self.critic_target([next_states, actions_next_states])
            y = rewards + self.gamma * Q_value_next_states * (1 - dones)

            Q_value_current_states = self.critic_main([states, actions])
            critic_loss = tf.reduce_mean(tf.square(y - Q_value_current_states))

        grads1 = tape1.gradient(critic_loss, self.critic_main.trainable_variables)
        self.opt_critic.apply_gradients(zip(grads1, self.critic_main.trainable_variables))

        # Actor loss
        with tf.GradientTape() as tape2:
            new_actions = self.actor_main(states)
            Q_value_current_states = self.critic_main([states, new_actions])
            actor_loss = - tf.reduce_mean(Q_value_current_states)

        grads2 = tape2.gradient(actor_loss, self.actor_main.trainable_variables)
        self.opt_actor.apply_gradients(zip(grads2, self.actor_main.trainable_variables))

        if self.train_step % self.replace_step == 0:
            self.update_target(tau=0.005)
        self.train_step += 1
