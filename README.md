Description: This repo shows the implementation of some RL algorithms. The algorithms (agents) are located in module. To use it, install the entire package via 'pip install -e.'. To be convenient, the training and testing are located in notebooks.

Algorithms tested: 
1. Deep-Q Network (DQN)
2. To be added ...

Games/environment tested:
1. Cart-pole
2. Lunar lander - discrete actions
3. Lunar lander - continuous actions
4. Bipedal walker - continuous actions
5. To be added ...

Dependencies: Refer to environment.yml to create the virtual environment. This file doesn't include gym since gym might be installed differently on different OS. So make sure you have gym installed correctly on your machine.

Hardware: Macbook Pro 16-inch M1 with CPU. GPU was not activated due to its inconsistent performance.

Discussion:

1) Deep-Q Network (DQN):
    a) Cart-pole: Easily achieve a score of 500 after about 15 minutes of training. If the training were performed longer, the score would be even higher.
