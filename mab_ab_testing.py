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
    def __init__(self, bandits, n_train=10000, n_test = 90000, band_type = 'gaussian'):
        
        if band_type=='gaussian':
            self.bandits = [GaussianBandit(bandit[0], bandit[1]) for bandit in bandits]
        else:
            self.bandits = [BernoulliBandit(bandit) for bandit in bandits]
            
        self.n_ads = len(bandits)
        self.total_reward=0
        self.N = np.zeros(self.n_ads)
        self.Q = np.zeros(self.n_ads)
        self.avg_rewards=[]
        self.n_train = n_train
        self.n_test = n_test
        
        self.eps = 0.1

    def train(self):
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
        
        print("After training the best performing ad is {}".format(best_ad_index+1))
        return best_ad_index
    
    def test(self, best_ad_index):
        for i in range(self.n_test):
            R = self.bandits[best_ad_index].get_reward()
            self.total_reward += R
            avg_reward_so_far = self.total_reward / (self.n_train + i + 1)
            self.avg_rewards.append(avg_reward_so_far)
        
        df = pd.DataFrame(self.avg_rewards, columns = ['A/B/n'])
        return df
    
    def epsilon(self):
        ad_chosen = np.random.randint(self.n_ads)
        
        for i in range(self.n_test):
            R = self.bandits[ad_chosen].get_reward()
            R = self.bandits[ad_chosen].get_reward()  # Observe reward
            self.N[ad_chosen] += 1
            self.Q[ad_chosen] += (1 / self.N[ad_chosen]) * (R - self.Q[ad_chosen])
            self.total_reward += R
            avg_reward_so_far = self.total_reward / (i + 1)
            self.avg_rewards.append(avg_reward_so_far)
            
            if np.random.uniform()<= self.eps:
                ad_chosen = np.random.randint(self.n_ads)
            else:
                ad_chosen = np.argmax(self.Q)
        
        best_ad_index = np.argmax(self.Q)
        print(self.Q)
        
        print("After epsilon the best performing ad is {}".format(best_ad_index+1))
        
        df = pd.DataFrame(self.avg_rewards, columns = ['A/B/n'])
        return df
    
    def plot(self, df):
        df['A/B/n'].plot()
        average_reward=np.round(sum(self.avg_rewards)/(self.n_train+self.n_test), 3)
        plt.title("Average reward is {}".format(average_reward))
        plt.xlabel("Number of Impressions")
        plt.ylabel("Reward")
        plt.show()


if __name__ =="__main__":
    
    # Gaussian Bandits
    # ads = [(3,2), (0,1), (-1,6), (1,4)]
    
    # Bernoulli Bandits
    ads = [0.004, 0.016, 0.2, 0.028, 0.031]
    
    def random_agent():
        ads_initialized = Agent(ads, band_type="gaussian1")
        best_ad=ads_initialized.train()
        df = ads_initialized.test(best_ad)
        ads_initialized.plot(df)
        
    def epsilon_greedy_agent():
        ads_initialized = Agent(ads, band_type="bernoulli")
        df = ads_initialized.epsilon()
        ads_initialized.plot(df)