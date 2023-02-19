import gym
import pandas as pd
import random
from collections import defaultdict

class OnPolicyMonteCarloPrediction(object):
    
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
            
        def epsilon_greedy_policy(self, state, Q):
            epsilon=.5
            if random.uniform(0,1) < epsilon:
                return self.env.action_space.sample()
            else:
                return max(list(range(self.env.action_space.n)), key=lambda x: Q[(state,x)])
            
        
        def generate_episode(self, Q):
            episode = []
            state = self.env.reset()

            for t in range(self.num_timesteps):
                action = self.epsilon_greedy_policy(self.return_state(state),Q)
                next_state, reward, done, info, _ = self.env.step(action)
                episode.append((state, action, reward))

                if done:
                    break
                state = next_state
            return episode
        
        def optimal_policy_identification(self):
            Q = defaultdict(float)
            total_return = defaultdict(float)
            N = defaultdict(float)

            for i in range(self.num_iterations):
                episode = self.generate_episode(Q)

                all_state_action_pairs = [(s,a) for (s,a,r) in episode]
                rewards = [r for (s,a,r) in episode]

                for t, (state, action,_) in enumerate(episode):
                    if not (state, action) in all_state_action_pairs[0:t]:
                        state_trans = self.return_state(state)
                        R = sum(rewards[t:])
                        total_return[(state_trans, action)] = total_return[(state_trans, action)] + R
                        N[(state_trans, action)] += 1
                        Q[(state_trans, action)] = total_return[(state_trans, action)] / N[(state_trans, action)]

            df = pd.DataFrame(Q.items(), columns=['state_action_pair', 'value'])
            return df

