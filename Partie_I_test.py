# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:23:08 2024

@author: shiwenli

Partie I

Tester les signaux 0, 2, 4, 6, 8, 10

"""

import random

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from keras.models import load_model

from sklearn.metrics import confusion_matrix

from model_actionsB import model_snr

def normalize_dqn(x):
    k=np.ceil(np.log2(np.max(np.abs(x))+10**(-8)))
    x_n = x * 2**(-k)
    return x_n  # Retourne x mis à l'échelle et la valeur de k

# 环境类，用于模拟信号片段的处理
class SignalCodingEnv:
    def __init__(self, signal_segments, num, n_action):
        self.signal_segments = signal_segments
        self.num_actions = n_action    # 可用的模型数量
        self.state_index = 0  # 当前状态索引
        self.nbw = num * 5
        self.N = 128
        #self.num = num
        self.snrs = np.zeros((self.nbw, self.num_actions))
        
        self.id_s = (self.state_index % 600) // 100
        self.num_phase = self.state_index // 600
        self.signal_index = self.state_index % 100
        self.state = self.signal_segments[self.id_s][self.num_phase][self.signal_index*N:(self.signal_index+1)*N]
        
        self.state_n = normalize_dqn(self.state)

    def reset(self):
        self.state_index = 0
        
        return self.state_n

    def step(self, action):
        signal = self.signal_segments[self.id_s]
        index = self.state_index
        index_s = self.signal_index
        #state = self.state
        self.snrs[index, action] = model_snr(signal, index_s, action, self.num_phase)
                       
        self.state_index = (self.state_index + 1) % self.nbw
        #self.state_index = 0
        self.id_s = (self.state_index % 600) // 100
        self.num_phase = self.state_index // 600
        self.signal_index = self.state_index % 100
        
        done = self.state_index == 0  # 循环完成，回到初始状态
        self.state = self.signal_segments[self.id_s][self.num_phase][self.signal_index*N:(self.signal_index+1)*N]
        
        self.state_n = normalize_dqn(self.state)
        
        return self.state_n, self.snrs[index], done
    
    def action_numbers(self):
        return self.num_actions
    
    def state_num(self):
        return self.state_index


# Construction du réseau de neurones    
def build_mlp_model(input_size, hidden_size, output_size,loss='mse',optimizer=Adam()):
    # Réseau de type séquentiel
    model=Sequential()
    # Couche d'entrée
    model.add(Dense(units=hidden_size,input_dim=input_size, activation='relu'))
    # Couche intermédiaire
    model.add(Dense(units=hidden_size//2, activation='relu'))
    # Couche intermédiaire
    model.add(Dense(units=hidden_size//4, activation='relu'))
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
        history = self.model.fit(np.array([state]), np.array([y]), verbose=0)
        return history.history['loss'][0]   # Return the loss value

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
        
        #q_values = normalize_dqn(q_values)
        best_choix = np.zeros_like(q_values)
        index_max = np.argmax(q_values)
        best_choix[index_max] = 1 
        
        q_values = best_choix
        
        loss = self.update(state, q_values)
        
        return loss
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


# 训练DQN智能体
def train_dqn(env, model, episodes, update_period=10, epsilon=0.3, eps_decay=0.96):
    
    #rewards_history = []
    loss_history = []
    
    choix_actions = np.zeros(100*5)
    #initial_signal = env.reset()
    for episode in range(episodes):
        print(episode)
        
        initial_signal = env.reset()
        #choix_actions = []
        episode_losses = [] # List to store losses for the current episode
        
        done = False
        #total_reward = 0
        while not done:
            #action = dqn.act(initial_signal)
            
            # Stratégie epsilon-greedy 
            if episode == episodes - 1:
                q_values = model.predict(initial_signal)
                #print(q_values)
                action = np.argmax(q_values)
                
                #choix_actions[env.state_n] = action
            elif episode < 8:
                action = episode
            else:
                if  random.random() < epsilon:
                    action = random.choice(range(env.action_numbers()))
                else:
                    q_values = model.predict(initial_signal)
                    action = np.argmax(q_values)
            
            choix_actions[env.state_num()] = action
            
            next_state, reward, done = env.step(action)
            #total_reward += reward
            
            loss = model.train_step(initial_signal, action, reward, next_state, done)
            episode_losses.append(loss)
            
            if episode % update_period == 0:
                model.update_target_model()
            
            initial_signal = next_state

        # Mise à jour d'epsilon
        epsilon = max(epsilon*eps_decay, 0.001)
        
        average_loss = np.mean(episode_losses)  # Calculate average loss for the episode
        loss_history.append(average_loss)  # Append average loss to loss history
        
        #rewards_history.append(total_reward)
        #print(f"Episode {episode+1}/{episodes}, Total reward: {total_reward}")
    
        #print(choix_actions)
    
    return choix_actions, loss_history

# 保存模型函数
def save_model(model, filepath):
    model.save(filepath)
    print(f"Model saved to {filepath}")
    
# 加载模型函数
def load_trained_model(filepath):
    model = load_model(filepath)
    return model


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
'''
id_signal=1 #fault event from database
nb_w=100 # number of window per signals
nb_min=0 # start window in the signal

v1=DATA_S[id_signal][0]
v2=DATA_S[id_signal][1]
v3=DATA_S[id_signal][2]
i1=DATA_S[id_signal][3]
i2=DATA_S[id_signal][4]
i3=DATA_S[id_signal][5]


x=[v1[nb_min*N:N*(nb_min+nb_w)],v2[nb_min*N:N*(nb_min+nb_w)],v3[nb_min*N:N*(nb_min+nb_w)]] # creat input x of MMC
'''

x = []
nb_w=100 # number of window per signals
nb_min=0 # start window in the signal
for iden in range(6):
    id_signal = 2 * iden + 1
    
    v1=DATA_S[id_signal][0]
    v2=DATA_S[id_signal][1]
    v3=DATA_S[id_signal][2]
    i1=DATA_S[id_signal][3]
    i2=DATA_S[id_signal][4]
    i3=DATA_S[id_signal][5]
    x1=[v1[nb_min*N:N*(nb_min+nb_w)],v2[nb_min*N:N*(nb_min+nb_w)],v3[nb_min*N:N*(nb_min+nb_w)]]
    
    x.append(x1)
    
# Nombre d'états
n_state = 128

# Nombre d'actions
#n_action = env.action_space.n
n_action = 8

# Nombre d'épisodes
episodes = 300

# Nombre de couches cachées dans le réseau de neurones
n_hidden = 128

actions = []

  
env = SignalCodingEnv(x, nb_w, n_action)

# Initialisation des deux réseaux de neurones servant de modèles pour Q
#dqn1 = DQN_keras(build_mlp_model, n_state, n_hidden, n_action)

# 创建新的DQN实例
dqn = DQN_keras(build_mlp_model, n_state, n_hidden, n_action)

dqn.model.compile(optimizer=Adam(), loss='mse')

# Programme principal
#action_choix, loss_history = train_dqn(env, dqn, episodes)

# 保存训练后的模型
#save_model(dqn.model, "dqn_model2.h5")



# 加载保存的模型
trained_model = load_trained_model("dqn_model40bits.h5")
dqn.model = trained_model


# 生成时间刻度，从 0 到 2 秒，分为 nb_w 个点
time_stamps = np.linspace(0, 2, nb_w)

# Test

x3 = []
for iden2 in range(6):
    id_signal2 = 2 * iden2
    
    v1=DATA_S[id_signal2][0]
    v2=DATA_S[id_signal2][1]
    v3=DATA_S[id_signal2][2]
    i1=DATA_S[id_signal2][3]
    i2=DATA_S[id_signal2][4]
    i3=DATA_S[id_signal2][5]
    x4=[v1[nb_min*N:N*(nb_min+nb_w)],v2[nb_min*N:N*(nb_min+nb_w)],v3[nb_min*N:N*(nb_min+nb_w)]]
    
    x3.append(x4)

print('Test')

action_estimee = np.zeros(nb_w*6) 
action_best2 = np.zeros(nb_w*6) 
snr_best2 = np.zeros(nb_w*6) 
for i in range(nb_w*6):
    id_s = (i % 600) // 100
    num_phase = i // 600
    signal_index = i % 100
    state = x3[id_s][num_phase][signal_index*N:(signal_index+1)*N]
    
    state_n = normalize_dqn(state)
    
    q_values1 = dqn.predict(state_n)

    action1 = np.argmax(q_values1)
    
    #snr1 = model_snr(x3, id_s, action1, num_phase)
    
    action_estimee[i] = action1
    
    action_b = 0
    snr_b = 0
    for j in range(n_action):
        snr = model_snr(x3[id_s], signal_index, j, num_phase)
        if snr >= snr_b:
            snr_b = snr
            action_b = j
    action_best2[i] = action_b
    snr_best2[i] = snr_b

# 计算混淆矩阵
cm2 = confusion_matrix(action_best2, action_estimee, labels=[0, 1, 2, 3, 4, 5, 6, 7])

# 打印混淆矩阵
print("Confusion Matrix 2:")
print(cm2)

snr_estimee = np.zeros(nb_w*6)
for i in range(nb_w*6):
    id_s = (i % 600) // 100
    num_phase = i // 600
    signal_index = i % 100
    model = action_estimee[i]
    snr_estimee[i] = model_snr(x3[id_s], signal_index, model, num_phase)


for i in range(6):
    # 绘制散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(time_stamps, action_estimee[i*nb_w:(i+1)*nb_w], c='blue', marker='o')
    plt.title(f'Model Selection For Signal {2*i}')
    plt.xlabel('Time (s)')
    plt.ylabel('Selected Model')
    plt.xticks(np.linspace(0, 2, 11))
    plt.yticks(range(n_action + 1))  # 假设模型编号从 0 开始
    plt.grid(True)
    plt.show()
    
    # 绘制散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(time_stamps, action_best2[i*nb_w:(i+1)*nb_w], c='blue', marker='o')
    plt.title(f'Best Model Selection For Signal {2*i}')
    plt.xlabel('Time (s)')
    plt.ylabel('Selected Model')
    plt.xticks(np.linspace(0, 2, 11))  # 生成0到2秒的11个等间隔刻度
    plt.yticks(range(n_action + 1))  # 假设模型编号从 0 开始
    plt.grid(True)
    plt.show()
    
    # 创建一个新的图形
    plt.figure(figsize=(10, 6))
    plt.scatter(time_stamps, action_estimee[i*nb_w:(i+1)*nb_w], c='blue', marker='o', label='Estimated Model')
    plt.scatter(time_stamps, action_best2[i*nb_w:(i+1)*nb_w], c='red', marker='o', label='Best Model')
    plt.title(f'Model Selection Comparasion For Signal {2*i}')
    plt.xlabel('Time (s)')
    plt.ylabel('Selected Model')
    plt.xticks(np.linspace(0, 2, 11))  # 生成0到2秒的11个等间隔刻度
    plt.yticks(range(n_action + 1))  # 假设模型编号从 0 开始
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 创建一个新的图形
    plt.figure(figsize=(10, 6))
    plt.plot(time_stamps, snr_estimee[i*nb_w:(i+1)*nb_w], c='blue', marker='o', label='Estimated SNR')
    plt.plot(time_stamps, snr_best2[i*nb_w:(i+1)*nb_w], c='red', marker='o', label='Best SNR')
    plt.title(f'Estimate de SNR For Signal {2*i}')
    plt.xlabel('Time (s)')
    plt.ylabel('SNR (dB)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    snr_mean2 = np.mean(snr_estimee[i*nb_w:(i+1)*nb_w])
    print("Average SNR for estimated model is", snr_mean2)
    snr_bmean2 = np.mean(snr_best2[i*nb_w:(i+1)*nb_w])
    print("Average SNR for best model is", snr_bmean2)

