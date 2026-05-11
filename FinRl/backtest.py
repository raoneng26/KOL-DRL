import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
import matplotlib.dates as mdates
from stable_baselines3 import PPO
from envs.crypto_env import CryptoFusionEnv


# 【关键修改 1】：导入最新的双通道解耦网络，否则 load 模型时会找不到结构
from models.net import DualQueryRLFusion 


def calculate_metrics(net_worths, dates):
    """计算核心金融指标"""
    df_metrics = pd.DataFrame({'timestamp': dates, 'net_worth': net_worths})
    df_metrics['returns'] = df_metrics['net_worth'].pct_change().dropna()
    
    # 1. 累计收益率
    total_return = (df_metrics['net_worth'].iloc[-1] - df_metrics['net_worth'].iloc[0]) / df_metrics['net_worth'].iloc[0]
    
    # 2. 年化收益率 (假设数据是分钟级的)
    total_minutes = (df_metrics['timestamp'].iloc[-1] - df_metrics['timestamp'].iloc[0]).total_seconds() / 60
    if total_minutes == 0: total_minutes = 1
    annualized_return = (1 + total_return) ** ( (60 * 24 * 365.25) / total_minutes ) - 1
    
    # 3. 夏普比率 (假设无风险利率为 0，分钟级转年化)
    ann_factor = np.sqrt(60 * 24 * 365.25)
    if df_metrics['returns'].std() != 0:
        sharpe_ratio = ann_factor * (df_metrics['returns'].mean() / df_metrics['returns'].std())
    else:
        sharpe_ratio = 0
        
    # 4. 最大回撤 (MDD)
    df_metrics['roll_max'] = df_metrics['net_worth'].cummax()
    df_metrics['drawdown'] = (df_metrics['net_worth'] / df_metrics['roll_max']) - 1
    max_drawdown = df_metrics['drawdown'].min()
    
    # 5. 索提诺比率 (Sortino Ratio - 只考虑下行风险)
    downside_returns = df_metrics[df_metrics['returns'] < 0]['returns']
    if len(downside_returns) > 0 and downside_returns.std() != 0:
        sortino_ratio = ann_factor * (df_metrics['returns'].mean() / downside_returns.std())
    else:
        sortino_ratio = 0
        
    return {
        "Total Return": total_return * 100,
        "Annualized Return": annualized_return * 100,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown * 100,
        "Sortino Ratio": sortino_ratio
    }


def backtest():
    # ==========================================
    # 1. 数据加载与清洗
    # ==========================================
    df = pd.read_csv('./data/processed_finrl_data.csv')
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    if 'timestamp' in df.columns:
        df_vals = df.drop(columns=['timestamp'])
    else:
        df_vals = df

    df_vals.replace([np.inf, -np.inf], 0, inplace=True)
    df_vals.fillna(0, inplace=True)
    
    df_clean = df.copy()
    df_clean[df_vals.columns] = df_vals

    # ==========================================
    # 2. 提取测试集 (后 20%)
    # ==========================================
    train_size = int(len(df_clean) * 0.8)
    test_df = df_clean.iloc[train_size:].reset_index(drop=True)
    
    if len(test_df) == 0:
        raise ValueError("测试集为空！请检查数据长度或切分比例。")

    print(f"测试集时间范围: {test_df['timestamp'].iloc[0]} -> {test_df['timestamp'].iloc[-1]}")

    # ==========================================
    # 3. 加载模型与环境
    # ==========================================
    model_path = "./logs/best_model.zip" 
    print(f"正在加载最优双通道模型: {model_path}...")
    
    model = PPO.load(model_path)
    env = CryptoFusionEnv(test_df)
    obs, _ = env.reset()
    done = False
    
    # ==========================================
    # 4. 执行回测 (【修复】：合并成单一严密循环)
    # ==========================================
    net_worths =[env.initial_balance]
    dates = [test_df['timestamp'].iloc[0]]
    actions =[]
    
    step_idx = 0

    while not done and step_idx < len(test_df) - 1:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        net_worths.append(env.net_worth)
        actions.append(action)
        step_idx += 1
        dates.append(test_df['timestamp'].iloc[step_idx])

   
    actions_scalar = [int(a.item()) for a in actions]
    action_counts = collections.Counter(actions_scalar)
    
    print("\n 智能体的真实动作分布:", action_counts)
    print("0: 卖出, 1: 持有, 2: 买入\n")

    # ==========================================
    # 5. 计算基准与严格对齐
    # ==========================================
    initial_price = test_df.iloc[0]['close']
    if initial_price == 0: initial_price = 1e-6
    
    # 计算持有不动(Buy & Hold)的基准
    bnh_net_worths =[env.initial_balance * (price / initial_price) for price in test_df['close']]

    # 【关键修改 2】：强制截断对齐，防止绘图和计算时报维度不匹配错误
    min_len = min(len(dates), len(net_worths), len(bnh_net_worths))
    dates = dates[:min_len]
    net_worths = net_worths[:min_len]
    bnh_net_worths = bnh_net_worths[:min_len]

    rl_return = (net_worths[-1] - env.initial_balance) / env.initial_balance * 100
    bnh_return = (bnh_net_worths[-1] - env.initial_balance) / env.initial_balance * 100

    # 计算核心指标
    rl_metrics = calculate_metrics(net_worths, dates)
    bnh_metrics = calculate_metrics(bnh_net_worths, dates)

    # ==========================================
    # 6. 打印学术对照表 
    # ==========================================
    print("\n" + "="*60)
    print(f"{'Performance Metric':<25} | {'RL Agent (Dual-Query)':<20} | {'Benchmark (B&H)':<15}")
    print("-" * 60)
    for key in rl_metrics.keys():
        unit = "%" if "Return" in key or "Drawdown" in key else " "
        print(f"{key:<25} | {rl_metrics[key]:>14.2f}{unit} | {bnh_metrics[key]:>13.2f}{unit}")
    print("="*60 + "\n")

    
    # ==========================================
    # 7. 绘图 
    # ==========================================
    plt.figure(figsize=(14, 7), dpi=150) 
    
    plt.plot(dates, bnh_net_worths, label=f'Benchmark (Buy & Hold): {bnh_return:.2f}%', 
             color='grey', alpha=0.6, linestyle='--')
    
    # 加入了填色效果，让超额收益(Alpha)更加明显
    plt.plot(dates, net_worths, label=f'RL Agent (Ours): {rl_return:.2f}%', 
             color='#d62728', linewidth=2)
    
    # 如果 Agent 跑赢了基准，将中间的超额部分填充为浅红色
    plt.fill_between(dates, bnh_net_worths, net_worths, where=(np.array(net_worths) > np.array(bnh_net_worths)), 
                     color='#d62728', alpha=0.1, interpolate=True)

    plt.title('Out-of-Sample Backtest: Dual-Query RL Agent vs Benchmark', fontsize=16)
    plt.xlabel('Date (Year-Month-Day)')
    plt.ylabel('Net Worth (USDT)')
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    plt.xticks(rotation=45) 
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('final_backtest_dual_query.png')
    plt.show()

if __name__ == "__main__":
    backtest()