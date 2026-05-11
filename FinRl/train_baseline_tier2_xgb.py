import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from envs.crypto_env import CryptoFusionEnv

def prepare_xgb_data(env):
    """
    巧妙提取环境内已经处理好(标准化、对数收益化)的数据，确保 XGB 和 RL 特征一致
    """
    df = env.df.copy()
    
    # 构建监督学习的目标 (Target)：预测下一期的 real_close 是否大于当期
    # 1 代表涨 (应买入)，0 代表跌/平 (应空仓)
    df['target'] = (df['real_close'].shift(-1) > df['real_close']).astype(int)
    
    # 删除最后一行（因为没有下一期的 target）
    df = df.iloc[:-1]
    
    # 提取特征
    X_market = df[env.market_cols].values
    X_all = df[env.sent_cols + env.social_cols + env.market_cols].values
    y = df['target'].values
    
    return X_market, X_all, y

def train_and_eval_xgboost(train_env, val_env):
    print("正在提取并对齐特征...")
    X_mkt_train, X_all_train, y_train = prepare_xgb_data(train_env)
    X_mkt_val, X_all_val, y_val = prepare_xgb_data(val_env)
    
    # ---------------------------------------------------------
    # 模型 1：纯市场特征 XGBoost (Market-Only)
    # ---------------------------------------------------------
    print("\n训练 XGBoost (仅量价 Market 特征)...")
    clf_mkt = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
    clf_mkt.fit(X_mkt_train, y_train)
    y_pred_mkt = clf_mkt.predict(X_mkt_val)
    print("Market-Only 预测准确率:", accuracy_score(y_val, y_pred_mkt))
    
    # ---------------------------------------------------------
    # 模型 2：全模态特征 XGBoost (All Features)
    # ---------------------------------------------------------
    print("\n训练 XGBoost (全模态 Market + Text + Social)...")
    clf_all = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
    clf_all.fit(X_all_train, y_train)
    y_pred_all = clf_all.predict(X_all_val)
    print("All-Features 预测准确率:", accuracy_score(y_val, y_pred_all))
    
    # ---------------------------------------------------------
    # 回测引擎跑盘：把预测结果转化为动作丢入一致的环境
    # ---------------------------------------------------------
    def backtest_predictions(predictions, test_env, name=""):
        test_env.reset()
        net_worths =[test_env.initial_balance]
        
        for step in range(len(predictions)):
            pred = predictions[step]
            # 贪心预测规则：
            # 如果预测未来会涨 (1) -> 执行买入 (Action 2)。Env 内部做了限制，满仓时发 2 等于持有。
            # 如果预测未来会跌 (0) -> 执行卖出 (Action 0)。Env 内部做了限制，空仓时发 0 等于继续空仓。
            action = 2 if pred == 1 else 0
            
            obs, reward, done, truncated, info = test_env.step(action)
            net_worths.append(test_env.net_worth)
            if done: break
            
        # 补齐最后一步的环境结算
        if not done:
            test_env.step(1) # 最后一步持仓不动结算
            net_worths.append(test_env.net_worth)
            
        print(f"【第二梯队 - XGBoost {name}】最终净值: {test_env.net_worth:.2f} | 收益率: {((test_env.net_worth - test_env.initial_balance)/test_env.initial_balance)*100:.2f}%")
        return net_worths

    # 分别执行回测
    nw_mkt = backtest_predictions(y_pred_mkt, val_env, name="Market-Only")
    nw_all = backtest_predictions(y_pred_all, val_env, name="All-Features")
    
    return nw_mkt, nw_all

if __name__ == "__main__":
    df = pd.read_csv('./data/processed_finrl_data.csv')
    df.fillna(0, inplace=True)
    
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:].reset_index(drop=True)
    
    # 利用 Env 实例直接完成数据特征的一致性转换！
    train_env = CryptoFusionEnv(train_df)
    val_env = CryptoFusionEnv(val_df)
    
    train_and_eval_xgboost(train_env, val_env)