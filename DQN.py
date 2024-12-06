# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:35:33 2024

@author: shiwenli
"""

import random

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from model_actions import model_snr

def normalize_dqn(x):
    k=np.ceil(np.log2(np.max(np.abs(x))+10**(-8)))
    x_n = x * 2**(-k)
    return x_n  # Retourne x mis à l'échelle et la valeur de k

# 环境类，用于模拟信号片段的处理
class SignalCodingEnv:
    def __init__(self, signal_segments, num):
        self.signal_segments = signal_segments
        self.num_actions = 2    # 可用的模型数量
        self.state_index = 0  # 当前状态索引
        self.nbw = 50
        self.N = 128
        self.num = num
        self.snrs = np.zeros(self.num_actions)
        self.state = self.signal_segments[0][self.num*N:(self.num+1)*N]
        self.state_n = normalize_dqn(self.state)

    def reset(self):
        self.state_index = 0
        
        return self.state_n

    def step(self, action):
        signal = self.signal_segments
        index = self.num
        #state = self.state
        self.snrs[action] = model_snr(signal, index, action)
                       
        #self.state_index = (self.state_index + 1) % self.nbw
        self.state_index = 0
        
        done = self.state_index == 0  # 循环完成，回到初始状态
        self.state = self.signal_segments[0][self.num*N:(self.num+1)*N]
        self.state_n = normalize_dqn(self.state)
        
        return self.state_n, self.snrs, done
    
    def action_numbers(self):
        return self.num_actions


# Construction du réseau de neurones    
def build_mlp_model(input_size, hidden_size, output_size,loss='mse',optimizer=Adam()):
    # Réseau de type séquentiel
    model=Sequential()
    # Couche d'entrée
    model.add(Dense(units=hidden_size*2,input_dim=input_size, activation='relu'))
    # Couche intermédiaire
    model.add(Dense(units=hidden_size, activation='relu'))
    # Couche intermédiaire
    model.add(Dense(units=hidden_size//2, activation='relu'))
    # Couche de sortie
    model.add(Dense(units=output_size, activation='linear'))
    #model.add(Dense(units=output_size, activation='relu'))
    # Compilation du réseau
    model.compile(loss=loss, optimizer=optimizer)
    
    return model


# Implementation du Deep Q-learning (DQN) avec keras
class DQN_keras():
    # Initialisation
    def __init__(self, build_func, input_size,
                hidden_size, output_size):
        
        self.model=build_func(input_size, hidden_size, output_size)
        
        self.target_model=build_func(input_size, hidden_size, output_size)
        self.target_model.set_weights(self.model.get_weights())
        
    def update(self, state, y):
        """Update the weighs of the network 
           given a training sample
        args:
          state: the input to predict
          y: the label
        """
        self.model.fit(np.array([state]), np.array([y]), verbose=0)

    def predict(self, state):
        """ Calcul des Q values pour chaques actions à partir d'un état""" 
        return self.model(np.array([state])).numpy()[0]
                
    def train_step(self, state, action, reward, next_state, done, gamma=0):
        """ Mise à jour du modèle à partir d'une seule expérience """
        q_values = self.predict(state)
        #print(q_values)
        q_values_next = self.predict(next_state)
        
        #target = q_values.copy()
        target = reward.copy()
        q_values = reward.copy()
        if done:
            q_values[action] = target[action]
        else:
            q_values[action] = target[action] + gamma * np.max(q_values_next)
        
        # 将小于0的值替换为0到1之间的随机小数
        #target[target < 0] = np.random.uniform(0, 1, size=target[target < 0].shape)
        
        self.update(state, q_values)
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


# 训练DQN智能体
def train_dqn(env, model, episodes, update_period=10, epsilon=0.4, eps_decay=0.95):
    
    rewards_history = []
    initial_signal = env.reset()
    for episode in range(episodes):
        
        #initial_signal = env.reset()
        
        choix_actions = []
        
        done = False
        total_reward = 0
        while not done:
            #action = dqn.act(initial_signal)
            
            # Stratégie epsilon-greedy 
            if episode == episodes - 1:
                q_values = model.predict(initial_signal)
                print(q_values)
                action = np.argmax(q_values)
            else:
                if  random.random() < epsilon:
                    action = random.choice(range(2))
                else:
                    q_values = model.predict(initial_signal)
                    action = np.argmax(q_values)
            
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            model.train_step(initial_signal, action, reward, next_state, done)
            
            if episode % update_period == 0:
                model.update_target_model()
            
            initial_signal = next_state
            
            choix_actions.append(action)
        
        # Mise à jour d'epsilon
        epsilon = max(epsilon*eps_decay, 0.001)
        
        rewards_history.append(total_reward)
        #print(f"Episode {episode+1}/{episodes}, Total reward: {total_reward}")
    
    #print(choix_actions)
    
    return rewards_history, action


"""
# Open the data
"""

Delta_u=18.31055
Delta_i=4.314
loaded = np.load('DATA_S2.npz')    
DATA_S=loaded['DATA_S2']


for number in range(len(DATA_S)):
    
    DATA_S[number][0]*=Delta_u
    DATA_S[number][1]*=Delta_u
    DATA_S[number][2]*=Delta_u
    DATA_S[number][3]*=Delta_i
    DATA_S[number][4]*=Delta_i
    DATA_S[number][5]*=Delta_i

#print("shape database",np.shape(DATA_S))


N=128 # size of window
fn=50 # nominal frequency
fs=6400 # samples frequency
    
#initialisation inputs

id_signal=1 #fault event from database
nb_w=50 # number of window per signals
nb_min=0 # start window in the signal

v1=DATA_S[id_signal][0]
v2=DATA_S[id_signal][1]
v3=DATA_S[id_signal][2]
i1=DATA_S[id_signal][3]
i2=DATA_S[id_signal][4]
i3=DATA_S[id_signal][5]


x=[v1[nb_min*N:N*(nb_min+nb_w)],v2[nb_min*N:N*(nb_min+nb_w)],v3[nb_min*N:N*(nb_min+nb_w)]] # creat input x of MMC

# Nombre d'états
n_state = 128

# Nombre d'actions
#n_action = env.action_space.n
n_action = 2

# Nombre d'épisodes
episodes = 100

# Nombre de couches cachées dans le réseau de neurones
n_hidden = 128

actions = []

for i in range(nb_w):
    
    env = SignalCodingEnv(x, i)
    
    # Initialisation des deux réseaux de neurones servant de modèles pour Q
    #dqn1 = DQN_keras(build_mlp_model, n_state, n_hidden, n_action)
    
    # 创建新的DQN实例
    dqn = DQN_keras(build_mlp_model, n_state, n_hidden, n_action)
    
    dqn.model.compile(optimizer=Adam(), loss='mse')
    
    # Programme principal
    rewards_episodes, action_choix = train_dqn(env, dqn, episodes)
    actions.append(action_choix)
    
    print(action_choix)

# 创建索引列表
indices = list(range(nb_w))

plt.scatter(indices, actions)

plt.title('Actions Taken in Each Episode')
plt.xlabel('Episode Index')
plt.ylabel('Action Taken')

plt.show()