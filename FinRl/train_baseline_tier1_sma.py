import pandas as pd
import numpy as np
from envs.crypto_env import CryptoFusionEnv

def run_sma_crossover_baseline(data_path='./data/processed_finrl_data.csv'):
    # 1. 加载与切分数据 (保持与 RL 相同的切分)
    df = pd.read_csv(data_path)
    df.fillna(0, inplace=True)
    
    train_size = int(len(df) * 0.8)
    val_df = df.iloc[train_size:].copy().reset_index(drop=True)
    
    # ==========================================
    # 【修复点】：在这里计算均线时，使用原始的 'close'
    # ==========================================
    val_df['MA5'] = val_df['close'].rolling(window=5).mean()
    val_df['MA20'] = val_df['close'].rolling(window=20).mean()
    
    # 修复 Pandas 新版本关于 method='bfill' 的弃用警告
    val_df.bfill(inplace=True) 
    
    # 3. 初始化相同的环境，保证手续费、初始资金完全一致
    # 注意：把 val_df 传给 Env 后，Env 内部才会自动生成 'real_close' 用于算钱
    env = CryptoFusionEnv(val_df)
    obs, _ = env.reset()
    
    net_worths = [env.initial_balance]
    
    # 4. 模拟交易步进
    done = False
    step = 0
    while not done:
        # 获取当前的 MA 状态
        ma5 = val_df.iloc[step]['MA5']
        ma20 = val_df.iloc[step]['MA20']
        
        # 传统 SMA 交叉策略规则：
        # MA5 > MA20 时看多 (买入并持有) -> Action: 2 (Buy)
        # MA5 < MA20 时看空 (清仓不碰) -> Action: 0 (Sell)
        if ma5 > ma20:
            action = 2 
        elif ma5 < ma20:
            action = 0
        else:
            action = 1 # Hold
            
        # 注意：老版本 gym 返回4个值，新版本 gymnasium 返回5个值
        # 我们用 _ 忽略 info，确保兼容性
        obs, reward, done, truncated, info = env.step(action)
        
        net_worths.append(env.net_worth)
        step += 1
        
    print(f"【第一梯队 - SMA双均线策略】最终净值: {env.net_worth:.2f}")
    print(f"收益率: {((env.net_worth - env.initial_balance) / env.initial_balance) * 100:.2f}%")
    return net_worths

if __name__ == "__main__":
    run_sma_crossover_baseline()