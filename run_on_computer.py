#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 19:01:53 2019

@author: chenlimin
"""

# train on computer

import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, merge, Lambda, LSTM, Concatenate
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
import random
import matplotlib.pyplot as plt
from keras.layers import Lambda
import pickle

class StockTradingEnv:
    
    def __init__(self,df,init_amount):
        # record buging amount and selling amount
        self.balance = 0
        self.buy = 0
        self.sell = 0
        self.total = self.balance + self.buy + self.sell
        self.init_amount = init_amount
        # timestep for information/state index
        self.current_step = 0
    
        self.time_lag = 10
        # store the datadrame state info
        self.dataframe = df
    
    '''
    逻辑顺序： 先得到状态state info，在当日最后一刻再 make action --> step()
    
    设 current time = 30，则抽取 21-30天的十天数据，包括第30th天，在30th天最后一刻做决策
    '''
    
    def _nextObservation(self):
        # for simplicity
        t = self.current_step
    
        market_info = self.dataframe.iloc[t-self.time_lag+1:t+1].values
        account_info = [self.balance,self.buy,self.sell]
        state = (market_info,account_info)
        
        return state          
    
    def reset(self):
        
        self.balance = self.init_amount
        self.buy = 0 
        self.sell = 0
        self.total = self.balance + self.buy + self.sell
        
        #randomly sample a timestep
        self.current_step = np.random.randint(self.time_lag, 2000)
        # record the begining point for termination usage
        self.start = self.current_step
        
        return self._nextObservation()
    
    

    
    
    def account_update(self):
        
        # use close_return to update account_value
        return_rate = self.dataframe.loc[self.current_step, "return_Close"]
        
        self.buy += return_rate * self.buy
        self.sell -= return_rate * self.sell
        
        


    def step(self,action):
        
        today_value = self.buy + self.sell + self.balance
        # change sell bug account amount
        self._takeAction(action)
        
        # welcome the next day
        self.current_step += 1
        
        # change account based on the return price
        # diff of self.total is the reward
        self.account_update()
        
        next_day_value = self.buy + self.sell + self.balance
        
        # diff (before change_account, after change_account)
        reward = next_day_value - today_value
        
        next_state = self._nextObservation()
        
        # termination condition
        done = (self.balance + self.sell + self.buy < 0.5*self.init_amount) or \
        (self.current_step - self.start > 400)
        
        return next_state,reward,done
        

    
    def _takeAction(self,action):
        
        # change sell bug amount
        
        action_type = action[0]
        amount_rate = action[1]
        # 拿剩余资金的百分比ratio购买股票
        amount = amount_rate * self.balance
        '''
        买卖比例设计机制可能有很大问题，action的第二维度是当前balance比例的话其实没有意义，因为如果当前balance很微小的话就无意义了
        但是在当前obs中包含了 buy sell 的 情况下，理论上 bot 是可以被训练出来调整这些比率的
        可以暂时不改
        '''      
        # suppose action[0] 为 [0,1]间的连续数
                
        if action_type < 1/3:
            # buy
            
            # 先平仓
            self.balance += self.sell
            self.sell = 0
            # 再买
            self.balance -= amount
            self.buy += amount
        
        elif action_type > 2/3:
            # sell
            
            # 先平仓
            self.balance += self.buy
            self.buy = 0
            # 再卖
            self.balance -= amount
            self.sell += amount            
            
        else:
            # hold -- do nothing
            pass
            
        
    def render(self):
        
        profit = self.buy + self.sell + self.balance - self.init_amount
        
        # suppose initiate value is 10000, but for NN problem we hide it and only indirectly show it here
        real_init = 10000
        profit = real_init * profit/self.init_amount
        print('profit : {}'.format(str(profit)))    
        
        # also can include other account info
        return profit
        
    def action_sample(self):
        
        return random.random(),random.random()


class Buffer:
    
    def __init__(self):
        
        self.memory = []
        self.capacity = 1000000
        
    def add(self,experience):
        
        self.memory.append(experience)
        
        if len(self.memory) > self.capacity:
            self.memory = self.memory[-self.capacity:]

    def sample(self,n):
        n = min(n,len(self.memory))
        return random.sample(self.memory, n)
    
    
    def weighted_sample(self,sampling_size):
        # need sorted thus causing extra computing time
        pass

class Actor:
    
    def __init__(self,market_input_shape,account_input_shape):
        
        self.market_input_shape = market_input_shape
        self.account_input_shape = account_input_shape
        
        #self.action_space = action_space
        
        self.sess = tf.get_default_session()
        
        self.tau = 0.2
        
        self.main_actor_nn, self.main_trainable_weight, self.main_action, self.main_amount, self.market_input, self.account_input = self._built_actor_NN()
        self.target_actor_nn,_,_,_,_,_ = self._built_actor_NN()
        
        
        # tf.gradients 第3项中的 action_gradient 是干嘛的，权重？为什么加负号？？为什么不是双重tf.gradients()调用而是用权重？？
        self.action_gradient = tf.placeholder(tf.float32, [None, 2])
        # chain role, [grad_y*(dy/dx) for grad_y,y,x in (grad_ys,ys,xs)]
        # 梯度上升，所以加负号
        self.params_grad = tf.gradients(self.main_actor_nn.output, self.main_trainable_weight, -self.action_gradient)
        grads = zip(self.params_grad, self.main_trainable_weight) 
        # direct parameters update 
        self.optimize = tf.train.AdamOptimizer().apply_gradients(grads)
        
        
        
    def _built_actor_NN(self):
        
        # shape 还没写
        market_input = Input(shape=[self.market_input_shape[0],self.market_input_shape[1]],dtype=tf.float32,name = 'actor_market_input') 
        account_input = Input(shape=[self.account_input_shape],dtype=tf.float32,name = 'actor_account_input') 
        
        
        l1 = LSTM(32,return_sequences=True)(market_input)
        l2 = LSTM(8,return_sequences=False)(l1)
        # Concatenate 不知道参数设的对不对,因为格式语法不对，少加了一个()
        lstm_complete_input = Concatenate()([l2,account_input])
        
        h0 = Dense(64, activation='relu')(lstm_complete_input)
        h1 = Dense(8, activation='relu')(h0)
        
        # absoute value change layer
        # 是否应该 + 一层 batch normalization
        h2 = Lambda(lambda x : 40*x)(h1)
        action_type = Dense(1,activation='sigmoid')(h2)  
        
        amount_rate = Dense(1,activation='sigmoid')(h2)  
        
        # 可不可以不 merge,转化为 concat，不知道为啥要合并 ？？？
        output = Concatenate()([action_type,amount_rate])
        model = Model(input=[market_input,account_input],output=output)
        
        # return ction_type, amount_rate because we can then check through sess.run()
        return model, model.trainable_weights, action_type, amount_rate, market_input, account_input
    
    
    def main_predict(self,state):
        
        return self.main_actor_nn.predict(state)
        
    def target_predict(self,state):
        
        return self.target_actor_nn.predict(state)

    
    def train_main_NN(self,market_input,account_input,grads):
        
        self.sess.run(self.optimize, feed_dict={
            self.market_input: market_input,
            self.account_input: account_input,
            self.action_gradient: grads})

    def target_update(self):
        # extract main_nn weights
        main_weights = self.main_actor_nn.get_weights()
        target_weights = self.target_actor_nn.get_weights()
        # strange new target NN update way
        new_target_weight = list(map(lambda main,target:(main * self.tau + target * (1-self.tau)),main_weights,target_weights))
        
        self.target_actor_nn.set_weights(new_target_weight)


    def save_model(self,address):
        
        self.main_actor_nn.save(address)        
    
    def load_model(self):
        
        pass


class Critic:

    def __init__(self,market_input_shape,account_input_shape):
        
        self.market_input_shape = market_input_shape
        self.account_input_shape = account_input_shape
        
        self.tau = 0.2
        #self.action_space = action_space
        # 后期取gradients要用sess.run(feed_dict) 计算底层tf.gradients，需要feed这些变量
        self.main_critic_nn,self.action,self.market_input,self.account_input = self._built_critic_NN()
        self.target_critic_nn,_,_,_ = self._built_critic_NN()
        
        # 计算q value关于action的梯度,用于更新之后的 policy network梯度传导
        self.gradients = tf.gradients(self.main_critic_nn.output, self.action)
        
    def _built_critic_NN(self):
        
        # shape 还没写
        market_input = Input(shape=[self.market_input_shape[0],self.market_input_shape[1]],name = 'critic_market_input') 
        account_input = Input(shape=[self.account_input_shape],dtype=tf.float32,name = 'critic_account_input') 
        # new_input : action
        action = Input(shape=(2,),dtype=tf.float32,name='action')
        #return_sequences = True 后才能接 下一个lstm，不然就只有最后一个unit有输出，没法继续lstm多层叠加了ssssssss
        l1 = LSTM(32,return_sequences=True)(market_input)
        l2 = LSTM(8,return_sequences=False)(l1)
        # Concatenate 不知道参数设的对不对
        lstm_complete_input = Concatenate()([l2,account_input,action])
        
        h0 = Dense(64, activation='relu')(lstm_complete_input)
        h1 = Dense(32, activation='relu')(h0)
        h2 = Dense(8, activation='relu')(h1)
        
        Q =  Dense(1, activation='linear')(h2)       

        model = Model(input=[market_input,account_input,action],output=Q)
        optimizer = Adam()
        model.compile(optimizer=optimizer, loss='mse')
        
        # 返回 market_input,account_input 是因为之后为了获得gradient 要用 sess 
        # 和 feed_dict 来获得 model.output 对于 action 的 gradients
        return model,action,market_input,account_input
        
    def train_main_NN(self):
        
        pass

    
    def get_gradient(self,market_input,account_input,action,sess=None):
        if not sess:
            sess = tf.get_default_session()
        return sess.run(self.gradients,
                        feed_dict={self.market_input:market_input,
                                   self.account_input:account_input,
                                   self.action:action})[0]

    def target_update(self):
        # extract main_nn weights
        main_weights = self.main_critic_nn.get_weights()
        target_weights = self.target_critic_nn.get_weights()
        # strange new target NN update way
        new_target_weight = list(map(lambda main,target:(main * self.tau + target * (1-self.tau)),main_weights,target_weights))
        
        self.target_critic_nn.set_weights(new_target_weight)


    def save_model(self,address):
        
        self.main_critic_nn.save(address)
    
    def load_model(self):
        
        pass

def play_game(num_trajectory):
    
    for i in range(num_trajectory):
        
        # initate
        obs = env.reset()
        done = False

        while not done:
            experience = []
            # network sctructur concat 貌似要求 input分开输入
            market_input = obs[0]
            account_input = obs[1]
            
            market_input_shape = market_input.shape
            account_input_shape = len(account_input)
            
            # 为了使得维度 compatibale，修复bug从而作出的shape调整
            market_info = np.expand_dims(obs[0], axis=0)
            account_info = np.expand_dims(obs[1], axis=0)
            
            
            action = actor.main_predict([market_info,account_info])
            action = action[0]
            
            experience.append(obs)
            
            obs,reward,done = env.step(action)
            
            experience.append(action)
            experience.append(reward)
            experience.append(obs)
            
            buffer.add(experience)
        
        # render the performance
        profit = env.render()
        return profit


def experience_replay(sample_size):

    experience_batch = buffer.sample(sample_size)
    
    # extract info from memory into well - prepared batch
    for index,experience in enumerate(experience_batch):        
        
        state = experience[0]
        market_info = state[0]
        market_info = np.expand_dims(market_info, axis=0) # 左侧增加1维
        account_info = np.array(state[1])
        account_info = np.expand_dims(account_info, axis=0) # 左侧增加1维
        
        action = experience[1]
        action = np.expand_dims(action, axis=0) # 左侧增加1维
        
        reward = experience[2]
        reward = np.expand_dims(np.array([reward]), axis=0) # shape = (1,1)
        
        next_state = experience[3]
        next_market_info = next_state[0]
        next_market_info = np.expand_dims(next_market_info, axis=0) # 左侧增加1维
        next_account_info = next_state[1]
        next_account_info = np.expand_dims(next_account_info, axis=0) # 左侧增加1维
        
            
        if index == 0:
            
            batch_market = market_info
            batch_account = account_info
            
            batch_reward = reward
            batch_action = action
            
            batch_next_market = next_market_info
            batch_next_account = next_account_info
            
        else:
            
            batch_market = np.vstack([batch_market,market_info])
            batch_account = np.vstack([batch_account,account_info])
            
            batch_reward = np.vstack([batch_reward,reward])
            batch_action = np.vstack([batch_action,action])
            
            batch_next_market = np.vstack([batch_next_market,next_market_info])
            batch_next_account = np.vstack([batch_next_account,next_account_info])
            
        '''
        Keras 默认左侧含有batch_size，新维度为 None
        按左侧stack扩充维度即可即可
        '''

        # extract and get the batch of all these staff
        # feed them into the network
        
        '''
        # suppose we get
        batch_state --> batch_market + batch_account
        batch_action
        batch_reward
        batch_next_state --> batch_next_market + batch_next_account
        
        '''
    # we need to predict the next_state action
    # 应该是用 target NN predict (假设不使用 double Q - learning)
    batch_action_next_state = actor.target_actor_nn.predict([batch_next_market, batch_next_account])
    
    # 这里不可以用 double-Q learning，因为 DDPG 本来就是 discriministic，不需要选Q值最大的 action
    Q = critic.target_critic_nn.predict([batch_next_market,batch_next_account,batch_action_next_state])
    Q_label = Q + batch_reward 
    
    
    # train critic main NN
    critic.main_critic_nn.fit([batch_market,batch_account,batch_action],Q_label,epochs = 30, verbose=0)
    
    # batch_action = actor.model.predict([input_states,value_states]) 不需要了吧，直接memory记住action就行了吧
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        gradient_Q_to_action = critic.get_gradient(batch_market,batch_account,batch_action)
    
        actor.sess = sess
        # train actor main NN
        actor.train_main_NN(batch_market,batch_account,gradient_Q_to_action)
        
        # update target NN
        actor.target_update()
        critic.target_update()



# need load() function
if __name__ == '__main__':
  
  load = True
  # record profit
  record = []
  
  df = pd.read_csv('only_return.csv')
  init_amount = 0.0001
  env = StockTradingEnv(df,init_amount)

  # only simulate to get a shape parameter
  obs = env.reset()
  market_input = obs[0]
  account_input = obs[1]
  market_input_shape = market_input.shape
  account_input_shape = len(account_input)

  actor = Actor(market_input_shape,account_input_shape)
  critic = Critic(market_input_shape,account_input_shape)
  
  # load weights
  if load == True:
    public = ''
    actor.main_actor_nn.load_weights(public+'actor_weight.h5')
    actor.target_actor_nn.load_weights(public+'actor_weight.h5')
    critic.main_critic_nn.load_weights(public+'critic_weight.h5')
    critic.target_critic_nn.load_weights(public+'critic_weight.h5')

  buffer = Buffer()
  
  # trajectory 可以少一点，耗时间多且玩太多局记不住，突破 memory.capacity，大不了慢慢更新 
  num_trajectory = 1
  # sample_size 多一点，不然 默认32的batch_size 没有意义
  sample_size = 640

  general_train = 3000

  for i in range(general_train):

      print('general train {}'.format(str(i)))
      # play and collect memory
      profit = play_game(num_trajectory)
      # extract from memory and batchly train
      experience_replay(sample_size)
      
      if i%500 == 0:
        # save model
        #actor.save_model('actor_0.h5')
        #critic.save_model('critic_0.h5')
        actor.main_actor_nn.save_weights(public+'actor_weight.h5')
        critic.main_critic_nn.save_weights(public+'critic_weight.h5')

        #save_to_drive('actor_weight.h5')
        #save_to_drive('critic_weight.h5')

      # record = profit growing history
      record.append(profit)
      
      if i%100 == 0:
        # save the reward_record through [[ pickle ]]
        file = open("reward_record.pickle","wb")
        pickle.dump(record,file)
        file.close()
    
      #pickle_off = open("Emp.pickle","rb")
      #emp = pickle.load(pickle_off)
      #print(emp)  
      
  # plt.plot(record)



'''

问题： 

(1)是否改进 double-Q-learning [DDPQ 无 double Q learning 的必要，因为只产出一个Q]
(2)是否weighted sampling
(3)sampling size 和 batch size 混了？ 【目前已改】
(4)怎么加速训练（cpu的游戏遍历+如何快速找到正确趋势） 【难】
(5)epoch反正速度快，就可以多train一点 [尝试]
(6)每隔200步奇怪地报错，实在不行就用本地电脑CPU跑 【在本地尝试，应该没事】
(7) optimize.apply_gradient() 默认没有 batch ???
(8) 尝试 em AWS 隔夜20000次训练 [怎么保存 reward growing 是关键]
(9) 直接把 _ 在 env.step() 中去掉，不需要刻意去模仿 gym [已成功，不报错了]


'''


'''
仔细想想 trading env 有点像 model-based learning， 就像机器人控制一样，我们无法去控制干预大部分外界，
只能去预测外界的变化，就像呈现在我们眼前的股价一样

tf.train.Optimizer 博大精深，有时间仔细研究，可以解决上述(7)问题

'''









