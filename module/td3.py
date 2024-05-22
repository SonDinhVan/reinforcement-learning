import numpy as np
import random

import tensorflow as tf
from keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.initializers import RandomUniform

from collections import deque

# Addressing Function Approximation Error in Actor-Critic Methods - https://arxiv.org/abs/1802.09477
# Twin-Delayed DDPG - https://spinningup.openai.com/en/latest/algorithms/td3.html
# https://github.com/ChienTeLee/td3_bipedal_walker/blob/master/TD3_BipedalWalker.ipynb


class TD3Agent():
    def __init__(self, state_size, action_size,
                 lr_actor=0.001, lr_critic=0.001,
                 gamma=0.95, batch_size=64,
                 memory_size=10**6, training_start=1000,
                 min_action=-1, max_action=1,
                 noise_std=0.2, noise_boundary=0.5,
                 min_noise_std=0.05, noise_decay=0.995,
                 update_period=10) -> None:
        """
        Initialization
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory_size = memory_size

        self.memory = deque(maxlen=self.memory_size)
        self.training_start = training_start  # training start after this

        self.lr_actor = lr_actor  # learning rate
        self.lr_critic = lr_critic

        self.min_action = min_action  # minimum value of action
        self.max_action = max_action  # maximum value of action

        # 6 neural networks in total, 2 for actor and 4 for critic
        self.actor_eval = self.build_actor_network()  # actor evaluate network
        self.actor_target = self.build_actor_network()  # actor evaluate network

        self.critic_eval_1 = self.build_critic_network()  # 1st critic evaluate network
        self.critic_target_1 = self.build_critic_network()  # 1st critic target network

        self.critic_eval_2 = self.build_critic_network()  # 2nd critic evaluate network
        self.critic_target_2 = self.build_critic_network()  # 2nd critic target network

        # update the target networks to be identical with the main networks
        self.update_target(self.actor_eval, self.actor_target, tau=1)
        self.update_target(self.critic_eval_1, self.critic_target_1, tau=1)
        self.update_target(self.critic_eval_2, self.critic_target_2, tau=1)

        # optimizer
        self.opt_actor = tf.keras.optimizers.legacy.Adam(learning_rate=self.lr_actor)
        self.opt_critic = tf.keras.optimizers.legacy.Adam(learning_rate=self.lr_critic)

        self.train_step = 0
        self.update_period = update_period  # period to update the target networks

        self.noise_std = noise_std  # noise standard deviation
        # noise will be clipped within the noise_boundary before adding to
        # the action value
        self.noise_boundary = noise_boundary
        self.min_noise_std = min_noise_std
        self.noise_decay = noise_decay  # noise decay

    def build_actor_network(self):
        """
        The actor network
        """
        last_init = RandomUniform(minval=-0.003, maxval=0.003)

        inputs = Input(shape=(self.state_size,))
        out = Dense(400, activation="relu")(inputs)
        # out = BatchNormalization()(out)
        out = Dense(300, activation="relu")(out)
        # out = BatchNormalization()(out)
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
        # state_out = Dense(32, activation="relu")(state_input)

        # Action as input
        action_input = Input(shape=(self.action_size,))
        # action_out = Dense(32, activation="relu")(action_input)

        # Both are passed through separate layer before concatenating
        concat = Concatenate()([state_input, action_input])
        # concat = BatchNormalization()(concat)

        out = Dense(400, activation="relu")(concat)
        # out = BatchNormalization()(out)
        out = Dense(300, activation="relu")(out)
        # out = BatchNormalization()(out)
        outputs = Dense(1)(out)

        # Outputs single value for give state-action
        model = Model([state_input, action_input], outputs)

        return model

    def update_target(self, network, target_network, tau=0.005):
        """
        Copy the weights of the network to the target model using soft update.
        When tau = 1 -> hard update, e.g., copy the exact weights.
        """
        for (weight, target) in zip(network.weights, target_network.weights):
            # update the target values
            target.assign(weight * tau + target * (1 - tau))

    def act(self, state, use_noise=True, noise_label='Gaussian'):
        """
        Get the action according to the state.

        Args:
            use_noise (bool, optional): True if noise is applied into actions
                for exploration.
            noise_label (str, optional):
                - 'OUNoise' if using OUNoise
                - 'Gaussian' if using Gaussian noise
            If Parameter Noise is used, use_noise should be set to False.
            During testing, use_noise should be set to False.
        """
        actions = self.actor_eval(state)
        if use_noise:
            # we add noise for exploration
            if noise_label == "Gaussian":
                # Gaussian noise
                noise = np.random.normal(0, self.noise_std, (1, self.action_size))
            # elif noise_label == 'OUNoise':
            #     # OU noise
            #     noise = OUActionNoise(mean=np.zeros(self.action_size), std_deviation=float(self.noise_dev) * np.ones(1))

            # clip the noise within the boundary
            noise = np.clip(noise, -self.noise_boundary, self.noise_boundary)
            actions += noise
        # we clip it since it might be out of range after adding noise
        actions = np.clip(actions, self.min_action, self.max_action)

        return actions[0]

    def store_data(self, state, action, reward, next_state, done):
        """
        Store data into the memory.
        """
        if len(self.memory) == self.training_start:
            print("Collect enough samples, training starting")
        # Append the new data to the memory
        self.memory.append([state, action, reward, next_state, done])

    def learn(self):
        """
        Training using the minibatch sampled from memory.
        """
        if len(self.memory) < self.training_start:
            return
        # sample a minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        states, actions, rewards, next_states, dones = [tf.convert_to_tensor(x, dtype=tf.float32) for x in zip(*minibatch)]

        states = tf.squeeze(states)
        dones = tf.reshape(dones, shape=(-1, 1))
        rewards = tf.reshape(rewards, shape=(-1, 1))
        next_states = tf.squeeze(next_states)

        # actions for the next states
        noise = np.random.normal(0, self.noise_std, (self.batch_size, self.action_size))
        noise = np.clip(noise, -self.noise_boundary, self.noise_boundary)
        actions_next_states = tf.clip_by_value(
            self.actor_target(next_states) + noise, self.min_action, self.max_action
        )

        Q_value_next_states_1 = self.critic_target_1([next_states, actions_next_states])
        Q_value_next_states_2 = self.critic_target_2([next_states, actions_next_states])

        y = rewards + self.gamma * (1 - dones) * tf.math.minimum(Q_value_next_states_1, Q_value_next_states_2)

        with tf.GradientTape() as tape1:
            Q_value_current_states_1 = self.critic_eval_1([states, actions])
            critic_loss_1 = tf.reduce_mean(tf.square(y - Q_value_current_states_1))

        grads1 = tape1.gradient(critic_loss_1, self.critic_eval_1.trainable_variables)
        self.opt_critic.apply_gradients(zip(grads1, self.critic_eval_1.trainable_variables))

        with tf.GradientTape() as tape2:
            Q_value_current_states_2 = self.critic_eval_2([states, actions])
            critic_loss_2 = tf.reduce_mean(tf.square(y - Q_value_current_states_2))

        grads2 = tape2.gradient(critic_loss_2, self.critic_eval_2.trainable_variables)
        self.opt_critic.apply_gradients(zip(grads2, self.critic_eval_2.trainable_variables))

        if self.train_step % self.update_period == 0:
            # in TD3, the policy network is updated less frequent compared to
            # actor network to improve stability during trainign
            with tf.GradientTape() as tape3:
                outputs = self.actor_eval(states)
                Q_values_1 = self.critic_eval_1([states, outputs])
                actor_loss = - tf.reduce_sum(Q_values_1)

            grads3 = tape3.gradient(actor_loss, self.actor_eval.trainable_variables)
            self.opt_actor.apply_gradients(zip(grads3, self.actor_eval.trainable_variables))

            # update the target networks using soft update
            self.update_target(self.actor_eval, self.actor_target, tau=0.005)
            self.update_target(self.critic_eval_1, self.critic_target_1, tau=0.005)
            self.update_target(self.critic_eval_2, self.critic_target_2, tau=0.005)

        self.train_step += 1
