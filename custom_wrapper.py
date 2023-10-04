import gym

class custom_wrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.previous_time = None
        self.previous_wins = None

        self.move_queue = []

        # Special move sequences (just considering the directional input)
        self.special_moves = [
            [[5], [6], [7]],  # Quarter-circle forward (Right)
            [[1], [2], [3]],  # Quarter-circle forward (Left)
            [[5], [4], [3]],  # Quarter-circle backward (Right)
            [[1], [8], [7]],  # Quarter-circle backward (Left)
            [[5], [1], [5]],  # Forward, Back, Forward
            [[1], [5], [1]],  # Back, Forward, Back
        ]  

        print("Applying Custom Reward Wrapper")

    def step(self, action):
        obs, reward, done, info, *_ = self.env.step(action)


        # Time-based reward
        if self.previous_time is not None:
            time_reward = obs["timer"][0] - self.previous_time
            if time_reward >= 0:
                time_reward = 0
            reward += time_reward / 100
        self.previous_time = obs["timer"][0]

        # Win-based reward
        if self.previous_wins is not None:
            win_reward = (obs["own_wins"][0] - self.previous_wins) 
            reward += win_reward * 50
        self.previous_wins = obs["own_wins"][0]

        # current_move = obs["action"]

        # # Initialize special_move_bonus
        # special_move_bonus = 0

        # # Sliding window of last 3 elements
        # sliding_window = []

        # for element in current_move:
        #     sliding_window.append(element)
        #     if len(sliding_window) > 3:  # Maintain window size of 3
        #         sliding_window.pop(0)

        #     if len(sliding_window) == 3:  
        #         for special_move in self.special_moves:
        #             if all(move == segment for move, segment in zip(special_move, sliding_window)):
        #                 special_move_bonus = .01
        #                 break  


        # # Add the bonuses to the reward
        # reward += special_move_bonus

        return obs, reward, done, info, *_

    def reset(self, seed=None):
        obs = self.env.reset()
        self.previous_time = None
        return obs


