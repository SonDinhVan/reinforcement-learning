This repository showcases the implementation of various Reinforcement Learning (RL) algorithms. The algorithms, or agents, are organized within modules for ease of use. To get started, install the entire package via pip: `pip install -e.`
For convenience, both training and testing are in Jupyter notebooks.

**Algorithms Implemented**
1. Deep-Q Network (DQN)
2. Deep Deterministic Policy Gradient (DDPG)
3. Twin Delay DDPG (TD3)

**Games/Environments Tested**
1. Cart-pole
2. Lunar Lander (discrete and continuous actions)
3. Bipedal Walker (Hardcore = False)
4. To be added...

**Dependencies**

Refer to the environment.yml file to create the virtual environment. Note that `gym` is not included in this file, as its installation method may vary across different operating systems. Ensure that you have `gym` installed correctly on your machine.

**Hardware Used**

The experiments were conducted on a MacBook Pro 16-inch M1 with CPU. The GPU was not activated due to its inconsistent performance.

**Discussion**

**1. Deep-Q Network (DQN)** \
**a) Cart-pole:** \
Achieved a score of > 1000 after about **10 minutes** of training.
Further training would likely result in even higher scores.

<img src="docs/dqn_cartpole/dqn_cartpole.gif" width="400"/>  <img src="docs/dqn_cartpole/dqn_cartpole_score_over_episodes.png" width="400"/> 

**b) Lunar Lander:** \
Achieved a score of > 250 after about **30 minutes** of training. \
Note: Sometimes, during training, the ship doesn't move, e.g., it will hover around one position for a long time, leading to a very long training. To deal with this, I stopped an episode when the number of time steps spending on landing exceeds a threshold, e.g., 1000 time steps.

<img src="docs/dqn_lunar_lander/lunar_lander.gif" width="400"/>  <img src="docs/dqn_lunar_lander/dqn_lunalander_loss_over_episodes.png" width="400"/> 

**2. Deep Deterministic Policy Gradient (DDPG)** \
DDPG is very unstable during training, and requires a very careful parameter tuning, especially those related to exploration, e.g., noise. I managed to get it run on Pendulum, Lunar Lander continuous, but not on Bipedal walker (YET).

**3. Twin Delayed DDPG (TD3)** \
TD3 shows a significant improvement compared to DDPG. \
**a) Bipedal Walker - Hardcore=False:** \
Achived a score of > 300 within **2600** episodes of training, **10 hours**. This takes long because I let `render_mode='human'`. In general, the walker learns how to balance within the first 500 episodes, knows how to walk after 1500 episodes, and start to improve from there. I stopped the training early, but it should solve the task.
Note: To encourage the walker walks faster, I shaped the reward function as:
`reward = reward + 10 * horizontal velocity`
This will punish the walker if he walks backward, and give him more reward if he walks forward.

<img src="docs/td3_bipedal_walker/td3_bipedal_walker_easy.gif" width="400"/>  <img src="docs/td3_bipedal_walker/td3_bipedal_walker_easy.png" width="400"/>


**b) Lunar Lander - continous=True:** \
Achieved a score of > 250 after about **40 minutes** of training, and around **550 episodes**. \
Note: To avoid the situation when the ship hovering at one spot, I punished the agent in each timestep as follows:
`reward = reward - 0.1 * 1.001**time_step`

<img src="docs/td3_lunar_lander/td3_lunarlander_continuous.gif" width="400"/>  <img src="docs/td3_lunar_lander/TD3_LunarLander_continuous.png" width="400"/>
