import gym

AYANE_COMBOS = [
    [2, 2, 2, 3],  # Renjin-Soryu-Sen
    [2, 2, 2, 1],  # Haijin
    [2, 2, 5, 2, 2],  # Renjin-Koeiso
    [2, 2, 5, 2, 3],  # Renjin-Yoen
    [2, 2, 3, 3, 3],  # Renjin-Ten-Ryugaku
    [2, 2, 3, 3, 7, 3],  # Renjin-Gurenbu
    [2, 2, 5, 3, 3],  # Renjin-Ryuso
    [2, 2, 5, 3, 7, 3],  # Renjin-Roso
    [2, 1, 2],  # Hasetsu
    [2, 3, 3],  # Hajin-Shinso
    [1, 2, 3],  # Rasen-Urajin
    [1, 2, 7, 3],  # Rasen-Urachi
    [7, 5, 2, 2],  # Fuzan-Ryubu
    [7, 5, 2, 3]  # Fuzan-Seppu
]


class custom_wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.previous_time = None
        self.previous_stage = None
        self.previous_own_wins = None
        self.special_moves = AYANE_COMBOS  
        self.sliding_window = []

    def step(self, action):
        obs, reward, done, info, *_ = self.env.step(action)

        current_move = obs["action"]

        # Initialize special_move_bonus
        special_move_bonus = 0

        for element in current_move:
            self.sliding_window.append(element)
            
            if len(self.sliding_window) > 6:  # Maintain window size of 6
                self.sliding_window.pop(0)

            if len(self.sliding_window) >= 3:  
                for special_move in self.special_moves:
                    if len(special_move) > len(self.sliding_window):
                        continue
                    
                    # Compare against the last len(special_move) elements in sliding_window
                    last_moves = self.sliding_window[-len(special_move):]
                    if all(move == segment for move, segment in zip(special_move, last_moves)):
                        special_move_bonus = .0025
                        break  

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
        return obs



