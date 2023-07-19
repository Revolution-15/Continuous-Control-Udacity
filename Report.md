# Continuous-Control-Udacity

## DRL - DDPG Method - Continuous Control

### Model Architecture
The Udacity provided DDPG code in PyTorch was used and adapted for this single agent (version 1) environment.

The algorithm uses two deep neural networks (actor-critic) with the following struture:
- Actor    
    - Hidden: (input, 128)  - ReLU
    - Hidden: (128, 128)    - ReLU
    - Output: (128, 4)      - TanH

- Critic
    - Hidden: (input, 128)              - ReLU
    - Hidden: (128+action_size, 128)    - ReLU
    - Output: (128, 1)                  - Linear


### Hyperparameters
- number of episodes = 500
- timesteps per episode = 1000
- epsilon = 1.0 at starting
- min value of epsilon = 0.01
- epsilon decay = 0.995
- Learning Rate: 1e-4 (in both DNN)
- Batch Size: 1024
- Replay Buffer: 1e5
- Gamma: 0.85
- Tau: 1e-3
- Noise parameters (0.15 theta and 0.2 sigma.)


## Results and Future Work
Environment solved in 344 episodes!! 
With average score of : 30.09


!(continuous_control.png)




The goal is to find perfect hyperparameters of the algorithms to have the best performance. Also need to explore different DRL algorithms which can solve the continuous control task.
