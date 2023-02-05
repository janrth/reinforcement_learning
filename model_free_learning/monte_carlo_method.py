import gym
import pandas as pd
from collections import defaultdict

class MonteCarloPrediction(object):
    
    def __init__(self,
                 env,
                 num_timesteps,
                 num_iterations):
        self.env = env
        self.num_timesteps = num_timesteps
        self.num_iterations = num_iterations
        
    def return_state(self, state):
        if len(state)==2:
            return state[0]
        else:
            return state
        
    def policy(self, state):
        if len(state)==2:
            return 0 if state[0][0] >= 17 else 1
        else:
            return 0 if state[0] >= 17 else 1
        
    def generate_episode(self):
        episode = []
        state = self.env.reset()

        for t in range(self.num_timesteps):
            action = self.policy(state)
            next_state, reward, done, info, _ = self.env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        return episode
    
    def value_function(self, every_visit=1):
        total_return = defaultdict(float)
        N = defaultdict(int)

        for i in range(self.num_iterations):
            episode = self.generate_episode()
            states, actions, rewards = zip(*episode)

            for t, state in enumerate(states):
                state_trans = self.return_state(state)
                if every_visit==1: 
                    R = (sum(rewards[t:]))
                    total_return[state_trans] = total_return[state_trans] + R
                    N[state_trans] = N[state_trans] + 1
                else:
                    if state not in states[0:t]:
                        R = (sum(rewards[t:]))
                        total_return[state_trans] = total_return[state_trans] + R
                        N[state_trans] = N[state_trans] + 1

        total_return = pd.DataFrame(total_return.items(),
                                    columns=['state', 'total_return'])
        N = pd.DataFrame(N.items(), columns=['state', 'N'])
        df = pd.merge(total_return, N, on='state')
        df['value'] = df.total_return/df.N
        return df