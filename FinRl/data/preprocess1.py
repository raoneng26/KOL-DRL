import pandas as pd
import numpy as np

# 1. 读取你包含 FinBERT 结果的源文件
INPUT_FILE = 'processed_data.csv' 
OUTPUT_FILE = 'processed_finrl_data.csv'

print(f"正在读取 {INPUT_FILE} ...")
df = pd.read_csv(INPUT_FILE)

# 2. 时间标准化 (核心步骤)
print("正在对齐时间戳...")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 【关键】将所有时间向下取整到分钟 (Floor to min)
# 这样 06:24:35 (有推文) 和 06:24:00 (有价格) 就会变成同一个时间点
df['timestamp'] = df['timestamp'].dt.floor('min')

# 3. 聚合合并 (Groupby)
# 这一步会将同一分钟内的多行数据（有的有推文，有的有价格）合并成一行
print("正在合并同一分钟的数据...")

# 定义聚合规则
agg_rules = {
    # 情感数据：取平均 (忽略 NaN)
    'sent_neu': 'mean',
    'sent_pos': 'mean',
    'sent_neg': 'mean',
    
    # 社交数据：取总和 (忽略 NaN)
    'retweets': 'sum',
    'likes': 'sum',
    'replies': 'sum',
    'gemo': 'mean', # 情感分通常取平均
    
    # 市场数据：取平均 (理论上同一分钟价格应该是一样的，或者取第一个非空值)
    'open': 'mean',
    'high': 'max',   # 最高价取最大
    'low': 'min',    # 最低价取最小
    'close': 'last', # 收盘价取最后
    'volume': 'sum'  # 成交量取和
}

# 仅对存在的列进行聚合，防止报错
valid_agg_rules = {k: v for k, v in agg_rules.items() if k in df.columns}

# 执行聚合
df_grouped = df.groupby('timestamp').agg(valid_agg_rules).reset_index()

# 4. 缺失值填充 (Filling)
print("正在填充缺失值...")

# A. 市场数据填充 (Forward Fill)
# 如果某分钟既没有推文也没有原始行情（数据断点），用上一分钟的价格填补
market_cols = ['open', 'high', 'low', 'close', 'volume']
existing_market_cols = [c for c in market_cols if c in df_grouped.columns]
df_grouped[existing_market_cols] = df_grouped[existing_market_cols].ffill().bfill()

# B. 情感/社交数据填充
# 如果某分钟有行情但没有推文，情感设为中性，热度设为0
if 'sent_neu' in df_grouped.columns:
    df_grouped['sent_neu'] = df_grouped['sent_neu'].fillna(1.0) # 默认为中性
    df_grouped['sent_pos'] = df_grouped['sent_pos'].fillna(0.0)
    df_grouped['sent_neg'] = df_grouped['sent_neg'].fillna(0.0)

social_cols = ['retweets', 'likes', 'replies', 'gemo']
for col in social_cols:
    if col in df_grouped.columns:
        df_grouped[col] = df_grouped[col].fillna(0)

# 5. 最终清洗与排序
print("正在进行最终清洗与排序...")

# 【关键】强制按时间排序！解决回测图乱线问题
df_grouped.sort_values('timestamp', ascending=True, inplace=True)

# 替换 Inf 为 0
numerical_cols = df_grouped.select_dtypes(include=[np.number]).columns
df_grouped[numerical_cols] = df_grouped[numerical_cols].replace([np.inf, -np.inf], 0)

# 再次检查 NaN
assert not df_grouped[numerical_cols].isnull().values.any(), "错误：数据中仍含有 NaN，请检查聚合逻辑！"

# 6. 保存
df_grouped.to_csv(OUTPUT_FILE, index=False)
print(f"✅ 修复完成！")
print(f"已保存至: {OUTPUT_FILE}")
print(f"最终行数: {len(df_grouped)}")
print(f"时间范围: {df_grouped['timestamp'].iloc[0]} -> {df_grouped['timestamp'].iloc[-1]}")