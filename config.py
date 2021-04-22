
env_name = 'MsPacman-v0'
frame_stack = 4

#################### worker.py ####################
lr = 1e-4
eps = 1e-3
grad_norm=40
batch_size = 512
learning_starts = 20000
save_interval = 1000
target_network_update_freq = 2500
gamma = 0.99
prioritized_replay_alpha = 0.6
prioritized_replay_beta0 = 0.4
forward_steps = 3  # n-step forward
training_steps = 100000
buffer_capacity = 524288 
max_episode_length = 16384
slot_capacity = 2048  # cut one episode to slots to improve the buffer utilization

#################### train.py ####################
num_actors = 16
base_eps = 0.4
alpha = 0.7
log_interval = 5

#################### test.py ####################
render = False
save_plot = True
test_epsilon = 0.05