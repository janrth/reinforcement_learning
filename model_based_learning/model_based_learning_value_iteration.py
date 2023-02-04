import numpy as np
import gym

class ValueIteration(object):
    
    def __init__(self,
                 env,
                 gamma,
                 threshold,
                 num_iterations):
        self.env=env
        self.threshold=threshold
        self.num_iterations=num_iterations
    
    def q_function(self,
                   value_table,
                   s,
              ):
        gamma=1
        q_values = [sum([prob*(r + gamma * value_table[s_]) for prob, s_, r, _ 
                             in self.env.P[s][a]]) for a in range(self.env.action_space.n)]
        return q_values

    def value_iteration(self):
        value_table = np.zeros(self.env.observation_space.n)
    
        for i in range(self.num_iterations):
            updated_value_table = np.copy(value_table)
        
            for s in range(self.env.observation_space.n):
                Q_values = self.q_function(updated_value_table,s)
                value_table[s] = max(Q_values)
            
            if (np.sum(np.fabs(updated_value_table - value_table)) <= self.threshold):
                break
        return value_table
        
    def extract_policy(self,
                       value_table):
    
        policy = np.zeros(self.env.observation_space.n)
    
        for s in range(self.env.observation_space.n):
            Q_values = self.q_function(value_table,s)
            policy[s] = np.argmax(np.array(Q_values))
        return policy
    