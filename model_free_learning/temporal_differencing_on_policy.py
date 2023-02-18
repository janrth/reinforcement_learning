import gym
import numpy as np
import pandas as pd
import random

class TDOnPolicy(object):
    
    def __init__(self,
                 env,
                 num_episodes, 
                 num_timesteps,
                 epsilon,
                 alpha,
                 gamma):
        
        self.env = env
        self.num_episodes = num_episodes
        self.num_timesteps = num_timesteps
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        

    def epsilon_greedy(self, state, Q):
        if random.uniform(0,1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return max(list(range(self.env.action_space.n)), key=lambda x: Q[(state,x)])

    def transform_state(self, state):
        if isinstance(state, int):
            return state
        else:
            return state[0]
        
        
        
    def policy_computation(self):
        Q = {}

        for s in range(self.env.observation_space.n):
            for a in range(self.env.action_space.n):
                Q[(s,a)]=0.0

        for i in range(self.num_episodes):
            s = self.env.reset()
            a = self.epsilon_greedy(self.transform_state(s), Q)

            for t in range(self.num_timesteps):
                s_, r, done, _, _ = self.env.step(a)
                a_ = self.epsilon_greedy(s_, Q)

                Q[((self.transform_state(s)),a)] += self.alpha*(r+self.gamma*Q[(s_,a_)]-Q[(self.transform_state(s),a)])

                s = s_
                a = a_

                if done:
                    break
        return Q
    
    
    
    def find_optimal_policy(self, Q):
        res = []

        for i in range(self.env.observation_space.n):
            res.append(list(Q.values())[(0)+(i*4):(4)+(i*4)].index(max(list(Q.values())[(0)+(i*4):(4)+(i*4)])))

            qq = {'states':np.arange(self.env.observation_space.n).tolist(),
                  'best_action':res}

        df = pd.DataFrame(qq)
        return df