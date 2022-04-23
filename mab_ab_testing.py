# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:55:03 2022

@author: NareshKumarD
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BernoulliBandit(object):
    def __init__(self, p):
        self.p = p
    
    def get_reward(self):
        reward = np.random.binomial(n=1, p=self.p)
        return reward

class GaussianBandit(object):
    def __init__(self, mean, stdev):
        self.mean = mean
        self.stdev = stdev
    
    def get_reward(self):
        reward = np.random.normal(self.mean, self.stdev)
        return np.round(reward, 1)
    
class Agent(object):
    def __init__(self, bandits, train_test_split=0.1, num_trails = 100000, band_type = 'gaussian'):
        
        if band_type=='gaussian':
            self.bandits = [GaussianBandit(bandit[0], bandit[1]) for bandit in bandits]
        
        else:
            self.bandits = [BernoulliBandit(bandit) for bandit in bandits]
            
        self.n_ads = len(bandits)
        self.n_train = int(num_trails*train_test_split)
        self.n_test = int(num_trails*(1-train_test_split))
        self.reward_dict = {}
        self.n_trails = num_trails
            
    def reset_agent(self, optimistic = False):
        self.total_reward=0
        self.avg_rewards=[]
        
        if optimistic == False:
            self.Q = np.zeros(self.n_ads)
            self.N = np.zeros(self.n_ads)
            
        else :
            self.Q = np.ones(self.n_ads)*10
            self.N = np.ones(self.n_ads)

    def random_agent(self):
        
        self.reset_agent()
        print("Agent is reset")
        
        for i in range(self.n_train):
            ad_chosen = np.random.randint(self.n_ads)
            R = self.bandits[ad_chosen].get_reward()  # Observe reward
            self.N[ad_chosen] += 1
            self.Q[ad_chosen] += (1 / self.N[ad_chosen]) * (R - self.Q[ad_chosen])
            self.total_reward += R
            avg_reward_so_far = self.total_reward / (i + 1)
            self.avg_rewards.append(avg_reward_so_far)
        
        best_ad_index = np.argmax(self.Q)
        print(self.Q)
        
        print("Best performing ad after random actions is {}. Steps used to train: {}".format(best_ad_index+1, self.n_train))
        
        print("Using optimal ad for rest {} steps".format(self.n_test))
        for i in range(self.n_test):
            R = self.bandits[best_ad_index].get_reward()
            self.total_reward += R
            avg_reward_so_far = self.total_reward / (self.n_train + i + 1)
            self.avg_rewards.append(avg_reward_so_far)
        
        print("Average reward is: ", self.avg_rewards[-1])
        df = pd.DataFrame(self.avg_rewards, columns = ['A/B/n'])
        return df, best_ad_index, self.Q, self.N
    
    def optimistic_agent(self):
        
        self.reset_agent(optimistic = True)
        print("Agent is reset")
        
        ad_chosen = np.random.randint(self.n_ads)
        
        for i in range(self.n_trails):
            R = self.bandits[ad_chosen].get_reward()
            self.N[ad_chosen] += 1
            self.Q[ad_chosen] += (1 / self.N[ad_chosen]) * (R - self.Q[ad_chosen])
            self.total_reward += R
            avg_reward_so_far = self.total_reward / (i + 1)
            self.avg_rewards.append(avg_reward_so_far)
            
            ad_chosen = np.argmax(self.Q)
        
        best_ad_index = np.argmax(self.Q)
        print(self.Q)
        
        print("After optimistic method the best performing ad is {}.".format(best_ad_index+1))
        print("Average reward is: ", self.avg_rewards[-1])
        df = pd.DataFrame(self.avg_rewards, columns = ['A/B/n'])
        
        return df, best_ad_index, self.Q, self.N
    
    def epsilon_greedy(self, eps):
        
        self.reset_agent()
        print("Agent is reset")
        
        ad_chosen = np.random.randint(self.n_ads)
        
        for i in range(self.n_trails):
            R = self.bandits[ad_chosen].get_reward()
            self.N[ad_chosen] += 1
            self.Q[ad_chosen] += (1 / self.N[ad_chosen]) * (R - self.Q[ad_chosen])
            self.total_reward += R
            avg_reward_so_far = self.total_reward / (i + 1)
            self.avg_rewards.append(avg_reward_so_far)
            
            if np.random.uniform()<= eps:
                ad_chosen = np.random.randint(self.n_ads)
            else:
                ad_chosen = np.argmax(self.Q)
            
        best_ad_index = np.argmax(self.Q)
        print(self.Q)
        
        print("After epsilon-greedy method the best performing ad is {}. Epsilon: {}".format(best_ad_index+1, eps))
        
        print("Average reward is: ", self.avg_rewards[-1])
        
        self.reward_dict[eps] = self.avg_rewards
        
        df= pd.DataFrame.from_dict(self.reward_dict)
        return df, best_ad_index, self.Q, self.N
    
    def ucb1(self, c = 0.1):

        self.reset_agent()
        print("Agent is reset")
        
        ad_indices = np.array(range(len(self.bandits)))
        
        for t in range(1, self.n_trails+1):
            if any(self.N==0):
                ad_chosen = np.random.choice(ad_indices[self.N==0])
            else:
                uncertainity = np.sqrt(np.log(t)/self.N)
                ad_chosen = np.argmax(self.Q + c*uncertainity)
            
            R = self.bandits[ad_chosen].get_reward()
            self.N[ad_chosen] += 1
            self.Q[ad_chosen] += (1 / self.N[ad_chosen]) * (R - self.Q[ad_chosen])
            self.total_reward += R
            avg_reward_so_far = self.total_reward / t
            self.avg_rewards.append(avg_reward_so_far)
            
        best_ad_index = np.argmax(self.Q)
        print(self.Q)
        
        print("After UCB1 method the best performing ad is {}.".format(best_ad_index+1))
        print("Average reward is: ", self.avg_rewards[-1])
        df = pd.DataFrame(self.avg_rewards, columns = ['A/B/n'])
        
        return df, best_ad_index, self.Q, self.N
        
    
    def plot(self, df, mode = 'random'):
        df.plot()
        
        if mode == 'random' or mode == 'single epsilon' or mode == 'optimistic':
            plt.title("Average reward is {}".format(df.values[-1]))
    
        elif  mode == 'multiple epsilons':
            plt.xscale('log')
            
        plt.xlabel("Number of Impressions")
        plt.ylabel("Reward")
        plt.show()

if __name__ =="__main__":
    
    # Gaussian Bandits
    # ads = [(3,2), (0,1), (-1,6), (1,4)]
    
    # Bernoulli Bandits
    ads = [0.004, 0.016, 0.2, 0.028, 0.031]
    
    eps_values = [0.01, 0.1, 0.2]
    
    def play(agent = 'random', dist = 'bern'):
        
        if dist == 'bern':
            print("Actual maximum reward is {}".format(max(ads)))
            ads_initialized = Agent(ads, band_type="bernoulli")
        else:
            print("Actual maximum reward is {}".format(max(ads)[0]))
            ads_initialized = Agent(ads, band_type="gaussian")
        
        if agent == 'random':
            rewards, best_ad, Q_values, N = ads_initialized.random_agent()
            
        elif agent == 'single epsilon':
            rewards, best_ad, Q_values, N = ads_initialized.epsilon_greedy(eps_values[0])
        
        elif agent == 'multiple epsilons':
            for eps in eps_values:
                rewards, best_ad, Q_values, N = ads_initialized.epsilon_greedy(eps)
        
        elif agent == 'optimistic':
            rewards, best_ad, Q_values, N = ads_initialized.optimistic_agent()
        
        elif agent=='ucb1':
            rewards, best_ad, Q_values, N = ads_initialized.ucb1()
        
        ads_initialized.plot(rewards, mode = agent)
