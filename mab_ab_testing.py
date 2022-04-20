# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:55:03 2022

@author: NareshKumarD
"""
import numpy as np
import pandas as pd
import cufflinks as cf
import plotly.offline

class BernoulliBandit(object):
    def __init__(self, p):
        self.p = p
    
    def display_ad(self):
        reward = np.random.binomial(n=1, p=self.p)
        return reward
    
class Game(object):
    def __init__(self, bandits, n_train=1000, n_test = 90000):
        self.bandits = [BernoulliBandit(bandit) for bandit in bandits]
        self.n_ads = len(bandits)
        self.total_reward=0
        self.N = np.zeros(self.n_ads)
        self.Q = np.zeros(self.n_ads)
        self.avg_rewards=[]
        self.n_train = n_train
        self.n_test = n_test

    def train(self):
        for i in range(self.n_train):
            ad_chosen = np.random.randint(self.n_ads)
            R = self.bandits[ad_chosen].display_ad()  # Observe reward
            self.N[ad_chosen] += 1
            self.Q[ad_chosen] += (1 / self.N[ad_chosen]) * (R - self.Q[ad_chosen])
            self.total_reward += R
            avg_reward_so_far = self.total_reward / (i + 1)
            self.avg_rewards.append(avg_reward_so_far)
        
        best_ad_index = np.argmax(self.Q)
        print(self.Q)
        
        print("After training the best performing ad is {}".format(best_ad_index))
        return best_ad_index
    
    def test(self, best_ad_index):
        for i in range(self.n_test):
            R = self.bandits[best_ad_index].display_ad()
            self.total_reward += R
            avg_reward_so_far = self.total_reward / (self.n_train + i + 1)
            self.avg_rewards.append(avg_reward_so_far)
        
        df = pd.DataFrame(self.avg_rewards, columns = ['A/B/n'])
        return df
    
    def plot(self, df):
        cf.go_offline()
        cf.set_config_file(world_readable=True, theme="white")

        df['A/B/n'].iplot(title="A/B/n Test Avg. Reward: {}".format(sum(self.avg_rewards)/(self.n_train+self.n_test)),
        xTitle='Impressions',
        yTitle='Avg. Reward')

if __name__ =="__main__":
    
    ads = [0.004, 0.016, 0.2, 0.028, 0.031]
    ads_initialized = Game(ads)
    best_ad=ads_initialized.train()
    df = ads_initialized.test(best_ad)
    ads_initialized.plot(df)