from enum import Enum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sklearn.metrics.pairwise import euclidean_distances


class Action(Enum):
    LEFT = 0
    FORWARD = 1
    RIGHT = 2
    NONE = 3


class HPEnv(gym.Env):
    def __init__(self, seq):
        self.seq = seq.upper()
        self.seq_len = len(self.seq)
        self.actions = []

        observation_sequence = []
        for character in self.seq:
            if character == 'H':
                observation_sequence.append(0)
            elif character == 'P':
                observation_sequence.append(1)
        self.observation_sequence = np.array(observation_sequence, dtype=np.int64)

        self.reset()

        if len(self.seq) <= 2:
            raise ValueError("len(seq) must be > 2")

        # Note that the action space is 
        # 0 = Left, 1 = Forward, 2 = Right
        self.action_space = spaces.Discrete(3)
        # The observation space is the series of actions taken
        # 0 = Left, 1 = Forward, 2 = Right, 3 = No action,
        self.observation_space = spaces.Tuple((spaces.Box(low=0, high=3, shape=(self.seq_len,), dtype=np.int64),
                                               spaces.Box(low=0, high=1, shape=(self.seq_len,), dtype=np.int64)))

        # Whether the first left turn left has been taken
        # False after environment reset
        self.first_turn_left = False

    def step(self, action):
        if not type(action) is int:
            action = action.item()

        if not self.action_space.contains(action):
            raise ValueError("%r (%s) invalid" % (action, type(action)))

        # Force the first turning action to be Left
        if (action != 1) and (self.first_turn_left is False):
            if action == 2:
                action = 0
            self.first_turn_left = True

        def get_new_pos(action):
            prev_pos = self.state[-1]
            prev_prev_pos = self.state[-2]
            direction_order = [(0, 1), (1, 0), (0, -1), (-1, 0)] # up, right, down, left
            cur_direction = (prev_pos[0] - prev_prev_pos[0], prev_pos[1] - prev_prev_pos[1])
            cur_direction_idx = direction_order.index(cur_direction)
            if action == Action.LEFT.value:
                cur_direction_idx = (cur_direction_idx - 1) % 4  # Turn left 
            elif action == Action.RIGHT.value:
                cur_direction_idx = (cur_direction_idx + 1) % 4  # Turn right
            direction = direction_order[cur_direction_idx]
            new_pos = (prev_pos[0] + direction[0], prev_pos[1] + direction[1])
            return new_pos

        new_pos = get_new_pos(action)


        # Detects for collision
        if new_pos in self.state:
            # Try alternate action
            for _ in range(2):
                action = (action + 1) % 3
                new_pos = get_new_pos(action)
                if new_pos not in self.state:
                    break

        # Detects for being trapped
        is_free = False
        for next_action in (0, 1, 2):
            next_pos = get_new_pos(next_action)
            if next_pos not in self.state:
                is_free = True
        is_trapped = not is_free

        self.actions.append(action)
        self.state.append(new_pos)
        observation = self.observe()


        self.terminated = is_trapped
        self.truncated = len(self.state) == self.seq_len

        reward = self._compute_reward()
        # No reward if the entire sequence hasn't been laid out and isn't trapped
        if (len(self.actions) + 2) < len(self.seq) and is_free:
            reward = 0.
        info = {
            'chain_length'   : len(self.state),
            'seq_length'     : self.seq_len,
            'actions'        : self.actions,
            'state'          : self.state,
            'first_turn_left': self.first_turn_left,
            'is_trapped'     : is_trapped,
        }

        return (observation, reward, self.terminated, self.truncated, info)

    def _compute_reward(self) -> float:
        # The reward is the number of H-H non-sequential contacts
        num_contacts = 0.
        h_positions = {pos for idx, pos in enumerate(self.state) if self.seq[idx] == 'H'}

        # Check adjacent H-H interactions
        for x, y in h_positions:
            neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            for nx, ny in neighbors:
                # Second condition checks if non-sequential
                if (nx, ny) in h_positions and self.state.index((nx, ny)) not in {self.state.index((x, y)) - 1, self.state.index((x, y)) + 1}:
                    num_contacts += 1

        return num_contacts / 2  # Each pair is counted twice


    def observe(self):
        observation_actions = np.ones(shape=(self.seq_len,), dtype=np.int64) * 3

        for i, action in enumerate(self.actions):
            observation_actions[i + 2] = action

        return (observation_actions, self.observation_sequence)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        self.actions = []
        self.prev_reward = 0
        self.state = [ (0, 0), (0, 1) ]
        self.terminated = len(self.seq) == 2
        observation = self.observe()
        self.first_turn_left = False

        return observation, {"state": self.state}
