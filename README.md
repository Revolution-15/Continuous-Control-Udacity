

<img src="https://camo.githubusercontent.com/7ad5cdff66f7229c4e9822882b3c8e57960dca4e/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f766964656f2e756461636974792d646174612e636f6d2f746f706865722f323031382f4a756e652f35623165613737385f726561636865722f726561636865722e676966">


# DRL - DDPG - Reacher Continuous Control
Udacity Deep Reinforcement Learning Nanodegree Program - Reacher Continuous Control


### Observations:
- To run the project just execute the <b>main.py</b> file.
- There is also an .ipynb file for jupyter notebook execution.
- If you are not using a windows environment, you will need to download the corresponding <b>"Reacher"</b> version for you OS system. Mail me if you need more details about the environment <b>.exe</b> file.
- The <b>checkpoint.pth</b> has the expected average score already hit.


### Requeriments:
- tensorflow: 1.7.1
- Pillow: 4.2.1
- matplotlib
- numpy: 1.11.0
- pytest: 3.2.2
- docopt
- pyyaml
- protobuf: 3.5.2
- grpcio: 1.11.0
- torch: 0.4.1
- pandas
- scipy
- ipykernel
- jupyter: 5.6.0


## The problem:
- The task solved here refers to a continuous control problem where the agent must be able to reach and go along with a moving ball controlling its arms.
- It's a continuous problem because the action has a continuous value and the agent must be able to provide this value instead of just chose the one with the biggest value (like in discrete tasks where it should just say which action it wants to execute).
- The reward of +0.1 is provided for each step that the agent's hand is in the goal location, in this case, the moving ball.
- The environment provides 2 versions, one with just 1 agent and another one with 20 agents working in parallel.
- For both versions the goal is to get an average score of +30 over 100 consecutive episodes (for the second version, the average score of all agents must be +30).


## The solution:
- DDPG has been used for this project.
- Hyperparameter tuning is the main urdle in this project where you have to find the right configuration for the project.
- For the future, I am planning to implement the algorithms apart from DDPG.

- TO implement the code in the repository follow the instructions below :

  
Run continuous_control.ipynb file.

my_model.py contains the actor and critic networks.

my_agent.py contains the agent code.
