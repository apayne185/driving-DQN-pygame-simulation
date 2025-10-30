# Deep Reinforcement Learning for 2D Racing (DQN + Q-Table)

This repository contains a 2D driving simulation built with Pygame + Gymnasium, where an autonomous agent learns using radar sensor readings to race around a curved racetrack using RL.
Two RL approaches are explored:

- Deep Q-Network (DQN) — neural network agent using PyTorch
- Q-Table Learning — tabular RL baseline

The simulation environment is implemented as custom Gym environment (gym_race), and gameplay is rendered using Pygame. The environment supports real-time rendering for visualization and non-render mode for faster training.


### Repository Structure
```
├── gym_race/                 # Gym environment
│   └── envs/
│       ├── pyrace_2d.py
│       ├── race_env.py
│       └── utils.py
├── models_DQN_v01/           # Saved DQN models
│   ├── best_dqn_model.pth
│   └── dqn_model_0.pth
├── models_QT_v02/            # Saved QTable memory, tables
│   ├── memory_3500.npy
│   └── q_table_3500.npy
├── Pyrace_RL_QTable.py                    # Main RL training/testing script
├── Pyrace_performance_analysis.ipynb      #Training analysis notebook
└── *.png                     # Racing environment visual assets
```

<br>

## Environment Overview
The Pyrace-v1 environment simulates a top-down 2D vehicle navigating a track using:
* Ray based sensor inputs
* Discrete actions (accelerate, turn left/right, ...)
* Reward shaping for progress and collision penalties


### Algorithms Implmented
| Algorithm |	File | Description |
|-------------|---------|-------------|
| Deep QNetwork (DQN)	| Pyrace_RL_QTable.py |	NN approximates QValues using PyTorch |
| Q-Table RL	| Pyrace_RL_QTable.py (legacy section & saved tables) |	Baseline QLearning for comparison |


### Observation Space 

The state consists of 5 radar sensor distances, normalized within [0,10]. Sensors are angled across the front of the car, with higher values meaning further distances. 

```python
[dist_1, dist_2, dist_3, dist_4, dist_5]
```


### Action Space
| Action	| Effect |
|---------|---------|
| 0	| Accelerate | 
| 1	| Turn left |
| 2	| Turn right |
| 3	| Brake (available in core env) |

  
### Reward Structure 
The agent is encouraged to move foward & pass the checkpoint (full lap around track), and avoid walls. 

| Event	| Reward |
|---------|---------|
| Checkpoint progress	| + distance-based reward
| Crash	-10000  | + distance traveled |
| Lap complete |	+10000 bonus |

----------

## Running the Code 

### Install Dependencies 

```bash 
pip install -r requirements.txt
```

### Training Agent
Within Pyrace_RL_QTable.py, change this line of code:

``` python 
#simulate()
load_and_play("best", learning=True)
```
to: 

``` python  
simulate()
# load_and_play("best", learning=True)
```


### Run Trained Agent 
Run the load_and_play function (and turn training off) to run the previously (best) trained agent: 

``` python 
#simulate()
load_and_play("best", learning=False) 
```



### Performance Analysis (Pyrace_performance_analysis.ipynb)


