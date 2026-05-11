# train.py
import pandas as pd
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from models.net import DualQueryRLFusion 
from models.net import VanillaMLPFusion 
from models.net import LateFusionMLP
from models.net import SingleQueryRLFusion
from models.net import NoMaskDualQueryRLFusion
from envs.crypto_env import CryptoFusionEnv

LOG_DIR = "./logs_VanillaMLPFusion/"
os.makedirs(LOG_DIR, exist_ok=True)

if __name__ == "__main__":
    # 1. 加载数据
    df = pd.read_csv('./data/processed_finrl_data.csv')
    df.fillna(0, inplace=True) 
    
    # 时序数据切分必须保证按时间顺序
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    # 3. 创建环境
    env = DummyVecEnv([lambda: CryptoFusionEnv(train_df)])
    eval_env = DummyVecEnv([lambda: CryptoFusionEnv(val_df)])

    # 4. 设置回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=LOG_DIR, 
        log_path=LOG_DIR,             
        eval_freq=2000,               
        deterministic=True,
        render=False
    )

    # 5. 配置模型 (适配双通道输出的宽广特征)
    policy_kwargs = dict(
        features_extractor_class=VanillaMLPFusion,
        features_extractor_kwargs=dict(
            sent_dim=3, social_dim=4, market_dim=5, hidden_dim=64, dropout=0.2
        ),
        # 注意：普通的网络特征较少，net_arch 恢复正常的 [64, 64] 即可
        net_arch=dict(pi=[64, 64], vf=[64, 64]) 
    )

    # 推荐参数：RL 处理噪音极大的金融数据时，PPO 的 clip_range 和 ent_coef 很重要
    model = PPO(
        "MlpPolicy", 
        env, 
        policy_kwargs=policy_kwargs, 
        verbose=1,
        learning_rate=1e-4, # 金融RL起步不要用太小的lr，否则会陷入始终持有的局部最优
        n_steps=2048,
        batch_size=128,     # 增加 batch_size 以稳定金融噪音方向
        ent_coef=0.01,      # 鼓励探索，防止过早只做单边交易
        clip_range=0.2,
        tensorboard_log=LOG_DIR 
    )

    # 6. 开始训练
    print(" 开始训练... (请在终端运行 'tensorboard --logdir ./logs/' 查看曲线)")
    model.learn(total_timesteps=200000, callback=eval_callback) # 金融环境建议增加训练步数
    
    print(" 训练结束！最优模型已保存至 logs_VanillaMLPFusion/best_model.zip")