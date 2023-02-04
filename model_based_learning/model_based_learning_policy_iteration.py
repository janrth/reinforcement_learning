import numpy as np
import gym 

class PolicyIteration(object):
    
    def __init__(self,
                 env,
                 num_iterations,
                 threshold):
        self.env = env
        self.num_iterations = num_iterations
        self.threshold = threshold
        
    def q_function(self,
               value_table,
               s,
              ):
        gamma=1
        q_values = [sum([prob*(r + gamma*value_table[s_]) for prob, s_, r, _ 
                             in self.env.P[s][a]]) for a in range(self.env.action_space.n)]
        return q_values
    
    def value_function(self,
                   value_table,
                   s,
                   a,
                   ):
        gamma=1
        value_table = sum([prob*(r+gamma*value_table[s_]) for prob, s_, r, _ 
                             in self.env.P[s][a]])
        return value_table
    
    def compute_value_function(self,
                               policy 
                               ):
        gamma=1
    
        value_table = np.zeros(self.env.observation_space.n)
    
        for i in range(self.num_iterations):
            updated_value_table = np.copy(value_table)
        
            for s in range(self.env.observation_space.n):
                a = policy[s]
                value_table[s] = self.value_function(updated_value_table,
                                                s,
                                                a)
            if (np.sum(np.fabs(updated_value_table - value_table)) <= self.threshold):
                break
        return value_table
    
    def extract_policy(self,
                       value_table,
                   ):
        gamma=1
        policy=np.zeros(self.env.observation_space.n)

        for s in range(self.env.observation_space.n):
            Q_values = self.q_function(value_table,
                                       s)
            policy[s] = np.argmax(np.array(Q_values))
        return policy
    
    def policy_iteration(self):
        gamma=1
        policy = np.zeros(self.env.observation_space.n)

        for i in range(self.num_iterations):
            value_function = self.compute_value_function(policy)
            new_policy = self.extract_policy(value_function)
            if (np.all(policy==new_policy)):
                break
            policy=new_policy
        return policy
        