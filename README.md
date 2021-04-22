# Ape-X (Distributed Prioritized Experience Replay)
## introduction
An Implementation of [Distributed Prioritized Experience Replay](https://arxiv.org/pdf/1803.00933.pdf) (Horgan et al. 2018) DQN in PyTorch and Ray.

This implementation is for multi-core single machine, works for openai gym environment.

## How to train
First go to config.py to adjust parameter settings if you want.

Then run:
```
python3 train.py
```
## How to test
```
python3 test.py
```
you can also render the test result or plot the result.

## Work in progress
The GPU utilization is a little bit low, which I'm trying to fix it.






