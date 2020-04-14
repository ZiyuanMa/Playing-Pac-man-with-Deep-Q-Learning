
# deep q-learning
image_shape = (84, 84)

input_shape = (4, 84, 84)
grad_norm=10
batch_size=32
double_q=False
buffer_size=50000
exploration_fraction=0.1
exploration_final_eps=0.01
train_freq=4
learning_starts=10000
save_interval=50000
target_network_update_freq=1000
gamma=0.99
prioritized_replay=False
prioritized_replay_alpha=0.6
prioritized_replay_beta0=0.4
dueling=False
atom_num=1
min_value=-10
max_value=10
n_steps = 3