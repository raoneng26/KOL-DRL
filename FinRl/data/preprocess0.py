import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# ==========================================
# 1. 配置 FinBERT-Tone
# ==========================================
MODEL_NAME = "yiyanghkust/finbert-tone"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {MODEL_NAME} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

def get_finbert_sentiment(texts, batch_size=32):
    """批量计算 FinBERT 情绪概率 (N, 3)"""
    probs_list = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Inferencing FinBERT"):
        batch_text = texts[i : i + batch_size]
        # 处理空文本
        batch_text = [t if isinstance(t, str) else "" for t in batch_text]
        
        inputs = tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True, max_length=64)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Softmax 归一化为概率
            probs = F.softmax(outputs.logits, dim=-1)
            probs_list.append(probs.cpu().numpy())
            
    return np.vstack(probs_list)

# ==========================================
# 2. 处理数据
# ==========================================
# 读取原始数据
df = pd.read_csv('simulated_event_dataset_multiscale.csv')

# A. 计算 3维 情绪概率
print("正在计算 FinBERT 情绪概率...")
sentiment_probs = get_finbert_sentiment(df['text'].tolist())

# 将概率合并回 DataFrame
df['sent_neu'] = sentiment_probs[:, 0] # Neutral
df['sent_pos'] = sentiment_probs[:, 1] # Positive
df['sent_neg'] = sentiment_probs[:, 2] # Negative

# B. 确认其他列 (确保是数值型)
# 社交特征 (4维)
social_cols = ['retweets', 'likes', 'replies', 'gemo']
for col in social_cols:
    df[col] = df[col].fillna(0).astype(float)
    # 建议做一下简单的归一化 (Log1p)，因为点赞数跨度很大
    df[col] = np.log1p(df[col]) 

# 市场特征 (5维)
market_cols = ['open', 'high', 'low', 'close', 'volume']
# 简单的Pct Change处理或者归一化，这里保持原始值，交给后面网络处理
# 也可以在这里计算技术指标...

# C. 保存训练数据
# 我们只需要保留核心特征列 + 时间戳
final_cols = ['timestamp'] + ['sent_neu', 'sent_pos', 'sent_neg'] + social_cols + market_cols
df_processed = df[final_cols].copy()

df_processed.to_csv('processed_finrl_data.csv', index=False)
print(f"数据处理完成！保存至 processed_finrl_data.csv，维度: {df_processed.shape}")