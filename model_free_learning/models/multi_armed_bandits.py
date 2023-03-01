import gym_bandits
import gym
import numpy as np

class MultiArmedBandit(object):
    
    def __init__(self,
                 num_rounds,
                 env):
        self.num_rounds = num_rounds
        self.env = env
        
    def empty_lists(self):
        count = np.zeros(2)
        sum_rewards = np.zeros(2)
        Q = np.zeros(2)
        return count, sum_rewards, Q
        
    def epsilon_greedy(self, epsilon, Q):
        if np.random.uniform(0,1) < epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(Q)
        
    def run_epsilon_greedy_learning(self, epsilon):
        self.env.reset()
        
        count, sum_rewards, Q = self.empty_lists()
        for i in range(self.num_rounds):
            arm = self.epsilon_greedy(epsilon=epsilon, Q=Q)
            next_state, reward, done, info = self.env.step(arm)
            count[arm] += 1
            sum_rewards[arm] += reward
            Q[arm] = sum_rewards[arm] / count[arm]

        return Q
    
    def softmax(self, T, Q):
        denom = sum([np.exp(i/T) for i in Q])
        probs = [np.exp(i/T)/denom for i in Q]
        arm = np.random.choice(self.env.action_space.n, p=probs)
        return arm
    
    def run_softmax_learning(self, T):
        self.env.reset()
        
        count, sum_rewards, Q = self.empty_lists()
        for i in range(self.num_rounds):
            arm = self.softmax(T, Q)
            next_state, reward, donbe, info = self.env.step(arm)
            count[arm] += 1
            sum_rewards[arm] += reward
            Q[arm] = sum_rewards[arm] / count[arm]

            T = T*.99
        return Q
    
    def ucb(self, i, Q, count):
        ucb = np.zeros(2)
        if i < 2:
            return i
        else:
            for arm in range(2):
                ucb[arm] = Q[arm] + np.sqrt((2*np.log(sum(count))) / count[arm])
            return (np.argmax(ucb))
        
    def run_ucb(self):
        self.env.reset()
        
        count, sum_rewards, Q = self.empty_lists()
        for i in range(self.num_rounds):
            arm = self.ucb(i, Q, count)
            next_state, reward, donbe, info = self.env.step(arm)
            count[arm] += 1
            sum_rewards[arm] += reward
            Q[arm] = sum_rewards[arm]/count[arm]
        return Q
    
    def thompson_sampling(self, alpha, beta):
        samples = [np.random.beta(alpha[i]+1, beta[i]+1) for i in range(2)]
        return np.argmax(samples)
    
    def run_thompson_learning(self):
        self.env.reset()

        alpha = np.ones(2)
        beta = np.ones(2)
        count, sum_rewards, Q = self.empty_lists()
        for i in range(self.num_rounds):
            arm = self.thompson_sampling(alpha, beta)

            next_state, reward, done, info = self.env.step(arm)
            count[arm] += 1
            sum_rewards[arm] += reward
            Q[arm] = sum_rewards[arm]/count[arm]
            if reward==1:
                alpha[arm] = alpha[arm]+1
            else:
                beta[arm] = beta[arm]+1

        return Q
    
    