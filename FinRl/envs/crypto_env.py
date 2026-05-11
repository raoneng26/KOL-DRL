#  cypto_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class CryptoFusionEnv(gym.Env):
    def __init__(self, df, initial_balance=10000, transaction_cost_pct=0.001):
        super(CryptoFusionEnv, self).__init__()
        
        # 1. 拷贝数据，防止修改原始 DataFrame
        self.df = df.copy().reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        
        # 2. 保留真实的收盘价，用于数学结算 (绝对不能用标准化后的数据算钱)
        self.df['real_close'] = self.df['close']
        
        # ==========================================
        # 【核心修复 1：将绝对价格转化为平稳相对特征】
        # 计算每一笔事件相对于上一次事件的价格变化率
        # ==========================================
        self.df['obs_open'] = (self.df['open'] / self.df['close'].shift(1)) - 1
        self.df['obs_high'] = (self.df['high'] / self.df['close'].shift(1)) - 1
        self.df['obs_low'] = (self.df['low'] / self.df['close'].shift(1)) - 1
        self.df['obs_close'] = (self.df['close'] / self.df['close'].shift(1)) - 1
        
        # 交易量波动极大，使用对数收益率处理，并限制极端值防止爆表
        self.df['obs_vol'] = np.log1p(self.df['volume']) - np.log1p(self.df['volume'].shift(1))
        self.df['obs_vol'] = self.df['obs_vol'].clip(-3, 3) 
        
        self.df.fillna(0, inplace=True) # 第一行没有shift，填0
        
        # 定义新的特征列名 
        self.sent_cols = ['sent_neu', 'sent_pos', 'sent_neg'] # 3
        self.social_cols = ['retweets', 'likes', 'replies', 'gemo'] # 4
        # 使用标准化后的观测特征喂给神经网络！
        self.market_cols =['obs_open', 'obs_high', 'obs_low', 'obs_close', 'obs_vol'] # 5
        
        # ==========================================
        # 【核心修复 2：社交特征标准化】
        # 社交特征里 retweets/likes 虽已 log，但 gemo 可能量纲不同，统一缩放
        # ==========================================
        scaler = StandardScaler()
        self.df[self.social_cols] = scaler.fit_transform(self.df[self.social_cols])
        
        self.sent_dim = len(self.sent_cols)     
        self.social_dim = len(self.social_cols) 
        self.market_dim = len(self.market_cols) 
        
        total_dims = self.sent_dim + self.social_dim + self.market_dim + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_dims,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        return self._next_observation(), {}

    def _next_observation(self):
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
            
        obs = self.df.iloc[self.current_step]
        
        sent_data = obs[self.sent_cols].values
        social_data = obs[self.social_cols].values
        market_data = obs[self.market_cols].values # 现在全是安全的相对波动率特征了
        
        pos_state = np.array([1.0 if self.shares_held > 0 else 0.0])
        return np.concatenate((sent_data, social_data, market_data, pos_state)).astype(np.float32)

    def step(self, action):
        # 结算必须用真实价格！
        current_price = self.df.iloc[self.current_step]['real_close']
        prev_net_worth = self.net_worth
        
        # 标志位：记录这回合是否真正产生了交易手续费
        executed_trade = False
        
        if action == 0 and self.shares_held > 0: # Sell
            self.balance += self.shares_held * current_price * (1 - self.transaction_cost_pct)
            self.shares_held = 0
            executed_trade = True
        elif action == 2 and self.balance > 0: # Buy
            shares_to_buy = self.balance / (current_price * (1 + self.transaction_cost_pct))
            self.shares_held += shares_to_buy
            self.balance = 0
            executed_trade = True
            
        self.net_worth = self.balance + self.shares_held * current_price
        
        if self.current_step > 0:
            prev_price = self.df.iloc[self.current_step - 1]['real_close']
        else:
            prev_price = current_price
            
        # ==========================================
        # 超额收益奖励计算
        # ==========================================
        market_return = (current_price - prev_price) / (prev_price + 1e-8)
        agent_return = (self.net_worth - prev_net_worth) / (prev_net_worth + 1e-8)
        
        # Reward = 战胜大盘的超额部分
        reward = agent_return - market_return
        
        # 【核心修复 3：仅在真正执行交易时扣除行为惩罚】
        # 原来你写的是 if action==2 就扣分，导致即使满仓了再选2也扣分，这是逼迫网络崩溃的原因之一
        if executed_trade:
            reward -= 0.0001
            
        self.prev_net_worth = self.net_worth
        self.current_step += 1
        
        if self.current_step >= len(self.df) - 1:
            done = True
            self.current_step = len(self.df) - 1
        else:
            done = False
        
        obs = self._next_observation()
        if np.isnan(obs).any():
            obs = np.nan_to_num(obs)
            
        return obs, float(reward), done, False, {}