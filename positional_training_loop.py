import datetime
import os  # for creating directories
import pickle
import random
import sys
from collections import deque
from time import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Transformer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from envs import Action
from models import *

seq = "HHHPPHPHPHPPHPHPHPPH"
seed = 42
algo = "positional_gin_fully_connected"
num_episodes = 100_000

base_dir = f"./{datetime.datetime.now().strftime('%m%d-%H%M')}-"
config_str = f"{seq[:6]}-{len(seq)}-mer-{algo}-{seed}-{num_episodes}"
save_path = base_dir + config_str + "/"
writer = SummaryWriter(f"logs/{save_path}")

# whether to show or save the matplotlib plots
display_mode = "save"  # save for CMD, show for ipynb
if display_mode == "save":
    save_fig = True
else:
    save_fig = False

# create the folder according to the save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Redirect 'print' output to a file in python
# orig_stdout = sys.stdout
# f = open(save_path + 'out.txt', 'w')
# sys.stdout = f

# apply flush=True to every print function call in the module with a partial function
from functools import partial

print = partial(print, flush=True)

print("seq = ", seq)
print("seed = ", seed)
print("algo = ", algo)
print("save_path = ", save_path)
print("num_episodes = ", num_episodes)

max_steps_per_episode = len(seq)

learning_rate = 0.0005

mem_start_train = max_steps_per_episode * 50  # for memory.size() start training
TARGET_UPDATE = 100  # fix to 100

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
gamma = 0.98  # discount rate
batch_size = 32
train_times = 10  # number of times train was run in a loop

# capped at 50,000 for <=48mer
buffer_limit = int(min(50000, num_episodes // 10))  # replay-buffer size

print("##### Summary of Hyperparameters #####")
print("learning_rate: ", learning_rate)
print("BATCH_SIZE: ", batch_size)
print("GAMMA: ", gamma)
print("mem_start_train: ", mem_start_train)
print("TARGET_UPDATE: ", TARGET_UPDATE)
print("buffer_limit: ", buffer_limit)
print("train_times: ", train_times)
print("##### End of Summary of Hyperparameters #####")

# Exploration parameters
max_exploration_rate = 1
min_exploration_rate = 0.01

# render settings
show_every = num_episodes // 1000  # for plot_print_rewards_stats
pause_t = 0.0
# metric for evaluation
rewards_all_episodes = np.zeros(
    (num_episodes,),
    # dtype=np.int32
)
reward_max = 0
best_folds = []
# keep track of trapped SAW
num_trapped = 0

warmRestart = True
decay_mode = "exponential"  # exponential, cosine, linear
num_restarts = 1  # for cosine decay warmRestart=True
exploration_decay_rate = 5  # for exponential decay
start_decay = 0  # for exponential and linear
print(f"decay_mode={decay_mode} warmRestart={warmRestart}")
print(
    f"num_restarts={num_restarts} exploration_decay_rate={exploration_decay_rate} start_decay={start_decay}"
)


# Nov30 2021 add one more column of step_E
hp_depth = 2  # {H,P} binary alphabet
action_depth = 4  # 0,1,2,3 in observation_box
pos_depth = 2


def one_hot_state(observation_states, observation_sequence):
    sequence_length = observation_sequence.shape[-1]
    observation_states = torch.tensor(observation_states).float()
    num_laid_out = observation_states.size(-2)
    observation_states = F.pad(
        observation_states, (0, 0, 0, sequence_length - num_laid_out)
    )
    observation_states = observation_states / float(sequence_length)
    one_hot_sequence = F.one_hot(torch.from_numpy(observation_sequence), num_classes=2)
    one_hot_input = torch.cat([observation_states, one_hot_sequence], dim=-1).float()
    return one_hot_input


# NOTE: partial_reward Sep15 changed to delta of curr-prev rewards
env = gym.make("HPEnv_v0", seq=seq)

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
env.action_space.seed(seed)

# initial state/observation
# NOTE: env.state != state here
# env.state is actually the chain of OrderedDict([((0, 0), 'H')])
# the state here actually refers to the observation!
initial_state = env.reset()

print("initial state/obs:")
print(initial_state)

# Get number of actions from gym action space
n_actions = env.action_space.n
print("n_actions = ", n_actions)

network_choice = "GINModel"
row_width = pos_depth + hp_depth
col_length = len(seq)


if network_choice == "RNN_LSTM_onlyLastHidden":
    # config for RNN
    input_size = row_width
    # number of nodes in the hidden layers
    hidden_size = 256
    num_layers = 2

    print("RNN_LSTM_onlyLastHidden with:")
    print(
        f"inputs_size={input_size} hidden_size={hidden_size} num_layers={num_layers} num_classes={n_actions}"
    )
    # Initialize network (try out just using simple RNN, or GRU, and then compare with LSTM)
    q = RNN_LSTM_onlyLastHidden(
        input_size, hidden_size, num_layers, n_actions, device
    ).to(device)
    q_target = RNN_LSTM_onlyLastHidden(
        input_size, hidden_size, num_layers, n_actions, device
    ).to(device)
elif network_choice == "MambaModel":
    # config for RNN
    input_size = row_width
    # number of nodes in the hidden layers
    hidden_size = 64
    num_layers = 2

    print("MambaModel with:")
    print(
        f"inputs_size={input_size} hidden_size={hidden_size} num_layers={num_layers} num_classes={n_actions}"
    )
    # Initialize network (try out just using simple RNN, or GRU, and then compare with LSTM)
    q = MambaModel(input_size, hidden_size, num_layers, n_actions, device).to(device)
    q_target = MambaModel(input_size, hidden_size, num_layers, n_actions, device).to(
        device
    )
elif network_choice == "TransformerModel":
    # config for Transformer
    input_size = row_width
    # number of nodes in the hidden layers
    hidden_size = 128
    num_layers = 2

    print("TransformerModel with:")
    print(
        f"inputs_size={input_size} hidden_size={hidden_size} num_layers={num_layers} num_classes={n_actions}"
    )
    # Initialize network (try out just using simple RNN, or GRU, and then compare with LSTM)
    q = TransformerModel(
        input_size, hidden_size, num_layers, n_actions, device, mask=True
    ).to(device)
    q_target = TransformerModel(
        input_size, hidden_size, num_layers, n_actions, device, mask=True
    ).to(device)
elif network_choice == "GATModel":
    # config for Transformer
    input_size = row_width
    # number of nodes in the hidden layers
    hidden_size = 128
    num_layers = 2

    print("TransformerModel with:")
    print(
        f"inputs_size={input_size} hidden_size={hidden_size} num_layers={num_layers} num_classes={n_actions}"
    )
    # Initialize network (try out just using simple RNN, or GRU, and then compare with LSTM)
    q = GATModel(input_size, hidden_size, num_layers, n_actions, device, fully_connected=True).to(device)
    q_target = GATModel(input_size, hidden_size, num_layers, n_actions, device, fully_connected=True).to(
        device
    )
elif network_choice == "GINModel":
    # config for Transformer
    input_size = row_width
    # number of nodes in the hidden layers
    hidden_size = 128
    num_layers = 2

    print("TransformerModel with:")
    print(
        f"inputs_size={input_size} hidden_size={hidden_size} num_layers={num_layers} num_classes={n_actions}"
    )
    # Initialize network (try out just using simple RNN, or GRU, and then compare with LSTM)
    q = GINModel(input_size, hidden_size, num_layers, n_actions, device, fully_connected=True).to(device)
    q_target = GINModel(input_size, hidden_size, num_layers, n_actions, device, fully_connected=True).to(
        device
    )


q_target.load_state_dict(q.state_dict())

optimizer = optim.Adam(q.parameters(), lr=learning_rate)


class ReplayBuffer:
    """
    for DQN (off-policy RL), big buffer of experience
    you don't update weights of the NN as you run
    through the environment, instead you save
    your experience of the environment to this ReplayBuffer
    It has a max-size to fit in certain examples
    """

    def __init__(self, buffer_limit):
        self.buffer = deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            # a tuple that tells us what the state was
            # at a particular point in time
            # we store the current state, the action we chose,
            # the state we ended up in, and whether finished or not
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        # converting the list to a single numpy.ndarray with numpy.array()
        # before converting to a tensor
        s_lst = np.array(s_lst)
        a_lst = np.array(a_lst)
        r_lst = np.array(r_lst)
        s_prime_lst = np.array(s_prime_lst)
        done_mask_lst = np.array(done_mask_lst)

        return (
            torch.tensor(s_lst, device=device, dtype=torch.float),
            torch.tensor(a_lst, device=device),
            torch.tensor(r_lst, device=device),
            torch.tensor(s_prime_lst, device=device, dtype=torch.float),
            torch.tensor(done_mask_lst, device=device),
        )

    def size(self):
        return len(self.buffer)

    def save(self, save_path):
        """save in .pkl file"""
        with open(save_path, "wb") as handle:
            pickle.dump(self.buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file_path):
        """load a .pkl file"""
        with open(file_path, "rb") as handle:
            self.buffer = pickle.load(handle)


memory = ReplayBuffer(buffer_limit)

# time the experiment
start_time = time()


def ExponentialDecay(
    episode,
    num_episodes,
    min_exploration_rate,
    max_exploration_rate,
    exploration_decay_rate=5,
    start_decay=0,
):
    decay_duration = num_episodes - start_decay
    exploration_rate = max_exploration_rate
    if episode > start_decay:
        exploration_rate = min_exploration_rate + (
            max_exploration_rate - min_exploration_rate
        ) * np.exp(-exploration_decay_rate * (episode - start_decay) / decay_duration)
    return exploration_rate


def train(q, q_target, memory, optimizer, n_episode):
    """
    core algorithm of Deep Q-learning

    do this training once per evaluation of the environment
    run evaluation once and train X times
    """
    for i in range(train_times):
        # sample from memory, which is not from the most recent runs
        # but from all previous runs in the memory, so you can be
        # more sample efficient, because you continuously learn from
        # past situations
        # key advantage of Off-policy
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        # the torch size is [batch_size, rows, cols], ie batch_first
        # print("DQN train --> s.size = ", s.size())
        # print(s)
        # print("DQN train --> r.size = ", r.size())
        # print(r)

        # forward once to q
        q_out = q(s)
        q_a = q_out.gather(1, a)
        # forward another time for q_target
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        # calculate the target value
        # if environment is done, there is no future reward,
        # mask the final step reward with done_mask (0.0 if done else 1.0)
        target = r + gamma * max_q_prime * done_mask
        # L1 loss but smoothed out a bit
        loss = F.smooth_l1_loss(q_a, target)
        # we will try to improve on Q(s,a)
        # how well our Q-function is at guessing the future long-term rewards
        # Q_targ() is the target Q network, a 2nd NN to stablize training
        # Q(s,a) = R(s,a) + γ*Q_targ(s_prime)*done_mask
        optimizer.zero_grad()
        loss.backward()
        # clip the policy_net.parameters()
        # Dec05 2021 found it did not work...
        # for param in q.parameters():
        #     param.grad.data.clamp_(-1, 1)
        optimizer.step()
        if i == 0:
            # Log loss onto Tensorboard
            writer.add_scalar("Q-Network Loss", loss, n_episode)


for n_episode in tqdm(range(num_episodes)):
    epsilon = ExponentialDecay(
        n_episode,
        num_episodes,
        min_exploration_rate,
        max_exploration_rate,
        exploration_decay_rate=exploration_decay_rate,
        start_decay=start_decay,
    )

    # reset the environment
    # Initialize the environment and state
    s = env.reset()

    s = one_hot_state(s[1]["state"], s[0][1])

    done = False
    score = 0.0

    # whether to avoid F in the next step?
    avoid_F = False

    for step in range(max_steps_per_episode):
        # unsqueeze(0) adds a dimension at 0th for batch=1
        # i.e. adds a batch dimension
        a = q.sample_action(s.float().unsqueeze(0), epsilon)

        # take the step and get the returned observation s_prime
        (s_prime, r, terminated, truncated, info) = env.step(a)
        # update a to the actual action taken, since the environment might take a different one
        # to enforce the first turn left constraint and avoid collisions
        a = info["actions"][-1]
        done = terminated or truncated

        # Only keep first turn of Left
        # internal 3actionStateEnv self.last_action updated
        # a = env.last_action
        # print("internal 3actionStateEnv last_action = ", a)

        """
        for NN:
            "one-hot" --> return the one-hot version of the quaternary tuple
        """
        s_prime = one_hot_state(info["state"], s_prime[1])
        # state_E_col, step_E_col)

        # NOTE: done_mask is for when you get the end of a run,
        # then is no future reward, so we mask it with done_mask
        done_mask = 0.0 if done else 1.0
        memory.put((s, a, r, s_prime, done_mask))
        s = s_prime

        # Add new reward
        # NOTE: Sep15 update partial_reward to be delta instead of progress*curr_reward
        # NOTE: Sep19 update reward to be a tuple, and reward is 0 until done
        score += r

        # check if the last action ended the episode
        if done:
            break

    # eventually if memory is big enough, we start running the training loop
    # start training after 2000 (for eg) can get a wider distribution
    # print("memory.size() = ", memory.size())
    if memory.size() > mem_start_train:
        train(q, q_target, memory, optimizer, n_episode)

    # Update the target network, copying all weights and biases in DQN
    if n_episode % TARGET_UPDATE == 0:
        q_target.load_state_dict(q.state_dict())

    # Add current episode reward to total rewards list
    rewards_all_episodes[n_episode] = score
    # Add episodic reward onto Tensorboard
    writer.add_scalar("Reward (Episode)", score, n_episode)
    # Add window of max episodic reward over the last 200 episodes onto Tensorboard
    if n_episode > 200:
        writer.add_scalar(
            "Reward (Episode Windowed)",
            np.max(rewards_all_episodes[n_episode - 200 : n_episode]),
            n_episode,
        )
    # update max reward found so far
    if score > reward_max:
        print("found new highest reward = ", score)
        reward_max = score
        best_folds.append(info)
        print(info)

    if (n_episode == 0) or ((n_episode + 1) % show_every == 0):
        print(
            "Episode {}, score: {:.1f}, epsilon: {:.2f}, reward_max: {}".format(
                n_episode,
                score,
                epsilon,
                reward_max,
            )
        )
        print(
            f"\ts_prime: {s_prime[:3], s_prime.shape}, reward: {r}, done: {done}, info: {info}"
        )
    # move on to the next episode

print("Complete")
# for time records
end_time = time()
elapsed = end_time - start_time
print(elapsed)

# Save the rewards_all_episodes with numpy save
with open(f"{save_path}{config_str}-rewards_all_episodes.npy", "wb") as f:
    np.save(f, rewards_all_episodes)

# Save the best foldings
with open(f"{save_path}{config_str}-best_folds", "wb") as f:
    pickle.dump(best_folds, f, protocol=pickle.HIGHEST_PROTOCOL)
    # Print the best foldings
    print("Best foldings:")
    print(best_folds)

# Save the pytorch model
# Saving & Loading Model for Inference
# Save/Load state_dict (Recommended)
torch.save(q.state_dict(), f"{save_path}{config_str}-state_dict.pth")

env.close()
