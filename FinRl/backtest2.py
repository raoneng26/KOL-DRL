import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from stable_baselines3 import PPO
from envs.crypto_env import CryptoFusionEnv

# 导入所有的特征提取器网络
from models.net import (
    DualQueryRLFusion, 
    SingleQueryRLFusion, 
    NoMaskDualQueryRLFusion, 
    NoSocialDualQueryRLFusion,
    NoTextDualQueryRLFusion,
    VanillaMLPFusion, 
    LateFusionMLP,
    GatedCrossAttentionFusion
)

def calculate_metrics(net_worths, dates):
    df_metrics = pd.DataFrame({'timestamp': pd.to_datetime(dates), 'net_worth': net_worths})
    df_metrics.set_index('timestamp', inplace=True)
    
    # 1. 计算总收益和最大回撤 (基于全周期)
    total_return = (df_metrics['net_worth'].iloc[-1] / df_metrics['net_worth'].iloc[0]) - 1
    df_metrics['roll_max'] = df_metrics['net_worth'].cummax()
    max_drawdown = ((df_metrics['net_worth'] / df_metrics['roll_max']) - 1).min()
    
    # 2. 【核心修复】降频到日度 (Daily) 计算夏普比率
    daily_nw = df_metrics['net_worth'].resample('D').last().dropna()
    daily_returns = daily_nw.pct_change().dropna()
    
    # 假设无风险利率为 0 (加密市场惯例)
    if daily_returns.std() != 0:
        # 日频转年化，因子是 sqrt(365.25)
        sharpe_ratio = np.sqrt(365.25) * (daily_returns.mean() / daily_returns.std())
        
        downside_returns = daily_returns[daily_returns < 0]
        sortino_ratio = np.sqrt(365.25) * (daily_returns.mean() / downside_returns.std()) if len(downside_returns) > 0 else 0
    else:
        sharpe_ratio, sortino_ratio = 0, 0

    # 3. 计算真实年化收益率
    days = (df_metrics.index[-1] - df_metrics.index[0]).days
    if days < 1: days = 1
    annualized_return = (1 + total_return) ** (365.25 / days) - 1
    
    return {
        "Total Return (%)": total_return * 100,
        "Annual Return (%)": annualized_return * 100,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown (%)": max_drawdown * 100,
        "Sortino Ratio": sortino_ratio
    }

def prepare_xgb_data(env):
    """为 XGBoost 提取并在 Env 中处理对齐好的数据"""
    df = env.df.copy()
    # Target: 下一期的 real_close 是否大于当期的 real_close (1: 涨, 0: 跌)
    df['target'] = (df['real_close'].shift(-1) > df['real_close']).astype(int)
    df = df.iloc[:-1] # 丢掉最后一行没有 target 的数据
    
    # 提取特征
    X_market = df[env.market_cols].values
    X_all = df[env.sent_cols + env.social_cols + env.market_cols].values
    y = df['target'].values
    return X_market, X_all, y

def plot_single_curve(dates, net_worths, bnh_aligned, bnh_metrics, metrics, model_name, config):
    """抽离出来的单独画图函数"""
    plt.figure(figsize=(12, 6), dpi=150)
    plt.plot(dates, bnh_aligned, label=f'Benchmark (B&H): {bnh_metrics["Total Return (%)"]:.2f}%', color='grey', alpha=0.6, linestyle='--')
    plt.plot(dates, net_worths, label=f'{model_name}: {metrics["Total Return (%)"]:.2f}%', color=config["color"], linewidth=2)
    
    plt.fill_between(dates, bnh_aligned, net_worths, where=(np.array(net_worths) > np.array(bnh_aligned)), color=config["color"], alpha=0.1, interpolate=True)
    plt.title(f'Out-of-Sample Backtest: {model_name} vs Benchmark', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Net Worth (USDT)')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    save_name = f"plot_single_{model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}.png"
    plt.savefig(save_name)
    plt.close()
    print(f"   -> 已保存单独回测图: {save_name}")


def run_comprehensive_backtest():
    # ==========================================
    # 1. 实验配置大矩阵 (配置模型路径)
    # ==========================================
    experiment_configs = {
        # 把原来的 No-Mask 扶正为我们的主模型，
        "Ours (Dual-Query)": {
            "path": "./logs_NoMaskDualQueryRLFusion/best_model.zip", 
            "extractor": NoMaskDualQueryRLFusion, 
            "color": "#d62728", "linewidth": 2.5, "linestyle": "-"
        },
        # 原来那个表现一般的 Ours，变成消融实验里的 Masked 版本
        "Ours (w/ Market Dropout)": {
            "path": "./logs/best_model.zip", 
            "extractor": DualQueryRLFusion, 
            "color": "#8c564b", "linewidth": 1.5, "linestyle": "-"
        },
        # 证明双通道比单通道强
        "Ablation: Single-Query": {
            "path": "./logs_SingleQueryRLFusion/best_model.zip", 
            "extractor": SingleQueryRLFusion, 
            "color": "#ff7f0e", "linewidth": 1.5, "linestyle": "-"
        },
        "Ablation: No-Text": {
            "path": "./logs_NoTextDualQueryRLFusion/best_model.zip", 
            "extractor": NoTextDualQueryRLFusion, 
            "color": "#9467bd", "linewidth": 1.5, "linestyle": "--"
        },
        "Ablation: No-Social": {
            "path": "./logs_NoSocialDualQueryRLFusion/best_model.zip", 
            "extractor": NoSocialDualQueryRLFusion, 
            "color": "#bcbd22", "linewidth": 1.5, "linestyle": "--"
        },
        # 下面两个代表普通网络
        "Late Fusion MLP": {
            "path": "./logs_LateFusionMLP/best_model.zip", 
            "extractor": LateFusionMLP, 
            "color": "#2ca02c", "linewidth": 1.5, "linestyle": "-"
        },
        "Vanilla MLP": {
            "path": "./logs_VanillaMLPFusion/best_model.zip", 
            "extractor": VanillaMLPFusion, 
            "color": "#1f77b4", "linewidth": 1.5, "linestyle": "-"
        },
        "Gated CrossAttention": {
            "path": "./logs_GatedCrossAttentionFusion/best_model.zip", 
            "extractor": GatedCrossAttentionFusion, 
            "color": "#1f77b4", "linewidth": 1.5, "linestyle": "-"
        },

    }

    # ==========================================
    # 2. 数据加载与清洗 (需切分 Train / Test)
    # ==========================================
    df = pd.read_csv('./data/processed_finrl_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    df_vals = df.drop(columns=['timestamp']) if 'timestamp' in df.columns else df
    df_vals.replace([np.inf, -np.inf], 0, inplace=True)
    df_vals.fillna(0, inplace=True)
    df_clean = df.copy()
    df_clean[df_vals.columns] = df_vals

    train_size = int(len(df_clean) * 0.8)
    train_df = df_clean.iloc[:train_size].reset_index(drop=True)
    test_df = df_clean.iloc[train_size:].reset_index(drop=True)

    print(f" 数据加载完成！测试集时间范围: {test_df['timestamp'].iloc[0]} -> {test_df['timestamp'].iloc[-1]}\n")

    dates_full = test_df['timestamp'].tolist()
    all_curves = {}
    all_metrics =[]

    # ==========================================
    # 3. 计算 Benchmark: Buy & Hold
    # ==========================================
    dummy_env = CryptoFusionEnv(test_df)
    initial_price = test_df.iloc[0]['close'] if test_df.iloc[0]['close'] != 0 else 1e-6
    bnh_net_worths =[dummy_env.initial_balance * (price / initial_price) for price in test_df['close']]
    
    bnh_metrics = calculate_metrics(bnh_net_worths, dates_full)
    all_curves["Benchmark (B&H)"] = {"curve": bnh_net_worths, "color": "grey", "linewidth": 2.0, "linestyle": "--"}
    
    bnh_row = {"Model Architecture": "Benchmark (B&H)"}
    bnh_row.update(bnh_metrics)
    all_metrics.append(bnh_row)

    # ==========================================
    # 【新增】第一梯队：SMA Crossover Baseline
    # ==========================================
    print(f" 正在回测: Tier 1 - SMA Crossover...")
    sma_env = CryptoFusionEnv(test_df)
    sma_env.reset()
    
    test_df_sma = test_df.copy()
    test_df_sma['MA5'] = test_df_sma['close'].rolling(5).mean()
    test_df_sma['MA20'] = test_df_sma['close'].rolling(20).mean()
    test_df_sma.bfill(inplace=True)
    
    sma_net_worths = [sma_env.initial_balance]
    for step_idx in range(len(test_df) - 1):
        ma5 = test_df_sma.iloc[step_idx]['MA5']
        ma20 = test_df_sma.iloc[step_idx]['MA20']
        action = 2 if ma5 > ma20 else (0 if ma5 < ma20 else 1)
        _, _, done, _, _ = sma_env.step(action)
        sma_net_worths.append(sma_env.net_worth)
        if done: break
            
    sma_metrics = calculate_metrics(sma_net_worths, dates_full[:len(sma_net_worths)])
    all_curves["SMA Crossover"] = {"curve": sma_net_worths, "color": "#9467bd", "linewidth": 1.5, "linestyle": "-."}
    sma_row = {"Model Architecture": "SMA Crossover"}
    sma_row.update(sma_metrics)
    all_metrics.append(sma_row)
    plot_single_curve(dates_full[:len(sma_net_worths)], sma_net_worths, bnh_net_worths[:len(sma_net_worths)], bnh_metrics, sma_metrics, "SMA Crossover", {"color": "#9467bd"})

    # ==========================================
    # 【新增】第二梯队：XGBoost Baseline (All-Features)
    # ==========================================
    print(f" 正在训练并回测: Tier 2 - XGBoost (All Features)...")
    train_env = CryptoFusionEnv(train_df)
    test_env_xgb = CryptoFusionEnv(test_df)
    
    _, X_all_train, y_train = prepare_xgb_data(train_env)
    _, X_all_test, _ = prepare_xgb_data(test_env_xgb)
    
    # 训练 XGBoost
    clf_all = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
    clf_all.fit(X_all_train, y_train)
    y_pred_all = clf_all.predict(X_all_test)
    
    # XGBoost 回测
    test_env_xgb.reset()
    xgb_net_worths =[test_env_xgb.initial_balance]
    for step_idx in range(len(test_df) - 1):
        action = 2 if y_pred_all[step_idx] == 1 else 0
        _, _, done, _, _ = test_env_xgb.step(action)
        xgb_net_worths.append(test_env_xgb.net_worth)
        if done: break

    xgb_metrics = calculate_metrics(xgb_net_worths, dates_full[:len(xgb_net_worths)])
    all_curves["XGBoost (All Feat)"] = {"curve": xgb_net_worths, "color": "#e377c2", "linewidth": 1.5, "linestyle": "-."}
    xgb_row = {"Model Architecture": "XGBoost (All Feat)"}
    xgb_row.update(xgb_metrics)
    all_metrics.append(xgb_row)
    plot_single_curve(dates_full[:len(xgb_net_worths)], xgb_net_worths, bnh_net_worths[:len(xgb_net_worths)], bnh_metrics, xgb_metrics, "XGBoost (All Feat)", {"color": "#e377c2"})

    # ==========================================
    # 4. 第三、四梯队：自动化遍历评估 RL 模型
    # ==========================================
    for model_name, config in experiment_configs.items():
        if not os.path.exists(config["path"]):
            print(f" 找不到模型文件: {config['path']}, 跳过 {model_name}...")
            continue
            
        print(f" 正在回测: {model_name}...")
        
        custom_objects = {"features_extractor_class": config["extractor"]}
        model = PPO.load(config["path"], custom_objects=custom_objects)
        
        env = CryptoFusionEnv(test_df)
        obs, _ = env.reset()
        done = False
        
        net_worths =[env.initial_balance]
        dates = [test_df['timestamp'].iloc[0]]
        
        step_idx = 0
        while not done and step_idx < len(test_df) - 1:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
            net_worths.append(env.net_worth)
            step_idx += 1
            dates.append(test_df['timestamp'].iloc[step_idx])

        min_len = min(len(dates), len(net_worths), len(bnh_net_worths))
        dates = dates[:min_len]
        net_worths = net_worths[:min_len]
        bnh_aligned = bnh_net_worths[:min_len]
        
        all_curves[model_name] = {"curve": net_worths, "color": config["color"], "linewidth": config["linewidth"], "linestyle": config["linestyle"]}
        
        metrics = calculate_metrics(net_worths, dates)
        row = {"Model Architecture": model_name}
        row.update(metrics)
        all_metrics.append(row)
        
        plot_single_curve(dates, net_worths, bnh_aligned, bnh_metrics, metrics, model_name, config)

    # ==========================================
    # 5. 绘制综合对比净值图 (All-in-One Chart)
    # ==========================================
    print("\n 正在绘制综合对比净值图...")
    plt.figure(figsize=(14, 8), dpi=300) # 更高的 DPI，方便论文直接截图
    
    for name, data in all_curves.items():
        final_ret = (data["curve"][-1] - dummy_env.initial_balance) / dummy_env.initial_balance * 100
        label_text = f'{name} ({final_ret:+.2f}%)'
        
        # 将 Ours 模型放在图层最上面，其他模型在下面作为陪跑
        zorder = 10 if "Ours" in name else 5 
        
        plt.plot(dates_full[:len(data["curve"])], data["curve"], 
                 label=label_text, 
                 color=data["color"], 
                 linewidth=data["linewidth"], 
                 linestyle=data["linestyle"],
                 zorder=zorder)

    plt.title('Out-of-Sample Backtest Comparison Across All Baseline Tiers', fontsize=16, fontweight='bold')
    plt.xlabel('Timeline', fontsize=12)
    plt.ylabel('Cumulative Net Worth (USDT)', fontsize=12)
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # 整理并优化图例布局 (防止遮挡右侧走势)
    plt.legend(loc='upper left', fontsize=10, framealpha=0.9, edgecolor='black', ncol=2)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('plot_all_models_comparison.png')
    print("   -> 综合对比图已保存: plot_all_models_comparison.png")

    # ==========================================
    # 6. 打印完美对齐的学术基准表格
    # ==========================================
    df_results = pd.DataFrame(all_metrics).set_index("Model Architecture")
    
    print("\n" + "="*95)
    print(f"{'Model Architecture':<28} | {'Total Ret':>10} | {'Ann. Ret':>10} | {'Sharpe':>8} | {'Max DD':>10} | {'Sortino':>8}")
    print("-" * 95)
    for index, row in df_results.iterrows():
        print(f"{index:<28} | {row['Total Return (%)']:>9.2f}% | {row['Annual Return (%)']:>9.2f}% | {row['Sharpe Ratio']:>8.2f} | {row['Max Drawdown (%)']:>9.2f}% | {row['Sortino Ratio']:>8.2f}")
    print("="*95 + "\n")

    # 保存一份 CSV 以备论文插表使用
    df_results.to_csv("academic_metrics_table.csv")
    print(" 所有实验结果已保存！(包含了各个图表及 academic_metrics_table.csv)")

if __name__ == "__main__":
    run_comprehensive_backtest()