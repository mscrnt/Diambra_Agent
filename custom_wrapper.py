import gym
from combos import ayane

class custom_wrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.previous_time = None
        self.previous_wins = None
        self.previous_opp_health = None
        self.special_moves = ayane
        self.previous_stage = None

        print("Applying Custom Reward Wrapper")

    def step(self, action):
        obs, reward, done, info, *_ = self.env.step(action)

        # Stage-based reward
        if self.previous_stage is not None and obs["stage"][0] > self.previous_stage:
            reward += 25  # Big reward for advancing stages
        self.previous_stage = obs["stage"][0]

        # Time-based reward
        # if self.previous_time is not None:
        #     time_reward = obs["timer"][0] - self.previous_time
        #     if time_reward >= 0:
        #         time_reward = 0
        #     reward += time_reward / 1000
        # self.previous_time = obs["timer"][0]

        # # Win-based reward
        # if self.previous_wins is not None:
        #     win_diff = obs["own_wins"][0] - self.previous_wins
            
        #     # Check if agent advanced to the next stage
        #     if self.previous_wins == 2 and obs["own_wins"][0] == 0:
        #         win_diff = 0  # Counter reset due to advancing to the next stage, so no change in reward
            
        #     # Check if agent lost the match
        #     if self.previous_wins == 1 and obs["own_wins"][0] == 0:
        #         win_diff = 0  # Pentalty for losing the match
            
        #     reward += win_diff * 50

        # self.previous_wins = obs["own_wins"][0]

        # current_actions = obs["action"]

        # # Translate current_actions into tuples
        # translated_actions = list(zip(current_actions[::2], current_actions[1::2]))

        # # Initialize special_move_bonus
        # special_move_bonus = 0

        # # Check each window of up to 4 actions for special moves
        # for window_size in range(1, 5):  # window sizes from 1 to 4
        #     for start_idx in range(len(translated_actions) - window_size + 1):  # sliding the window
        #         window = translated_actions[start_idx:start_idx + window_size]
        #         for special_move, sequences in self.special_moves.items():
        #             if any(all(move == segment for move, segment in zip(sequence, window)) for sequence in sequences):
        #                 print(f"Special move {special_move} detected!")
        #                 # Check if opponent's health has decreased
        #                 if self.previous_opp_health is not None and obs['opp_health'][0] < self.previous_opp_health:
        #                     special_move_bonus = 10
        #                     print(f"Special move {special_move} hit detected!")
        #                 break

        # # Update previous opponent health
        # self.previous_opp_health = obs['opp_health'][0]

        # # Add the bonuses to the reward
        # reward += special_move_bonus

        return obs, reward, done, info, *_

    def reset(self, seed=None):
        obs = self.env.reset()
        self.previous_time = None
        self.previous_stage = None
        self.previous_wins = None
        self.previous_opp_health = None  
        return obs


