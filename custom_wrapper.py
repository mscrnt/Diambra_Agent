import gym
from combos import ayane

class custom_wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.previous_time = None
        self.previous_stage = None
        self.previous_own_wins = None
        self.special_moves = AYANE_COMBOS  
        self.previous_opp_health = None
        self.special_moves = ayane

    def step(self, action):
        obs, reward, done, info, *_ = self.env.step(action)

        current_actions = obs["action"]

        # Translate current_actions into tuples
        translated_actions = list(zip(current_actions[::2], current_actions[1::2]))

        # Initialize special_move_bonus
        special_move_bonus = 0

        # Check each window of up to 7 actions for special moves
        for window_size in range(1, 8):  # window sizes from 1 to 7
            for start_idx in range(len(translated_actions) - window_size + 1):  # sliding the window
                window = translated_actions[start_idx:start_idx + window_size]
                for special_move, sequences in self.special_moves.items():
                    if any(all(move == segment for move, segment in zip(sequence, window)) for sequence in sequences):
                        # Check if opponent's health has decreased
                        if self.previous_opp_health is not None and obs['opp_health'][0] < self.previous_opp_health:
                            special_move_bonus = 1
                            print(f"Special move {special_move} hit detected!")
                        break

        # Update previous opponent health
        self.previous_opp_health = obs['opp_health'][0]

        # Add the bonuses to the reward
        reward += special_move_bonus

        # Time-based reward
        if self.previous_time is not None:
            time_reward = obs["timer"][0] - self.previous_time
            if time_reward >= 0:
                time_reward = 0
            reward += time_reward / 100
        self.previous_time = obs["timer"][0]
        
        # Stage-based reward
        if self.previous_stage is not None and obs["stage"][0] > self.previous_stage:
            reward += 100  # Big reward for advancing stages
        self.previous_stage = obs["stage"][0]

        # Win-based reward
        if self.previous_own_wins is not None:
            win_diff = obs["own_wins"][0] - self.previous_own_wins
            if self.previous_own_wins == 2 and obs["own_wins"][0] == 0:
                win_diff = 0
            if self.previous_own_wins == 1 and obs["own_wins"][0] == 0:
                win_diff = 0 # Set to -1 for penalty
            reward += win_diff * 50
        self.previous_own_wins = obs["own_wins"][0]

        return obs, reward, done, info, *_

    def reset(self, seed=None):
        obs = self.env.reset()
        self.previous_time = None  # Reset previous_time
        self.previous_stage = None  # Reset previous_stage
        self.previous_opp_health = None 
        return obs



