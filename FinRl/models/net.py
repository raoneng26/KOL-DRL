# net.py
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class DualQueryRLFusion(BaseFeaturesExtractor):
    def __init__(self, observation_space, sent_dim=3, social_dim=4, market_dim=5, hidden_dim=64, dropout=0.2):
        # 最终输出的维度: context(64) + actor_feat(64) + critic_feat(64) + pos_state(1) = 193
        features_dim = (hidden_dim * 3) + 1 
        super(DualQueryRLFusion, self).__init__(observation_space, features_dim=features_dim)
        
        self.sent_dim = sent_dim
        self.social_dim = social_dim
        self.market_dim = market_dim
        
        # 1. 独立非线性编码器
        self.sent_encoder = nn.Sequential(nn.Linear(sent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.social_encoder = nn.Sequential(nn.Linear(social_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.market_encoder = nn.Sequential(nn.Linear(market_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        
        # 2. 【核心注入】双通道动态 Query 生成器
        # 专为 Actor (动作/短线方向) 服务
        self.query_gen_actor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        # 专为 Critic (价值/长线波动风险) 服务
        self.query_gen_critic = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        
        self.attn_layer = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, batch_first=True)
        
    def forward(self, observations):
        # 1. 拆解状态输入
        idx_sent = self.sent_dim
        idx_social = idx_sent + self.social_dim
        idx_market = idx_social + self.market_dim
        
        sent = observations[:, 0 : idx_sent]
        social = observations[:, idx_sent : idx_social]
        market = observations[:, idx_social : idx_market]
        pos_state = observations[:, idx_market:] 

        if torch.isnan(observations).any():
             observations = torch.nan_to_num(observations)

        h_sent = self.sent_encoder(sent)
        h_soc = self.social_encoder(social)
        h_mkt = self.market_encoder(market)
        
        # --- 继承之前的验证成果：不对称模态Dropout (30%) ---
        # 强迫 RL 智能体在探索阶段去关注 Text 和 Social，防止其在 Market 维度上“偷懒”
        if self.training:
            # 生成 (Batch, 1) 的 mask
            mask = (torch.rand(h_mkt.size(0), 1, device=h_mkt.device) > 0.30).float()
            h_mkt = h_mkt * mask

        # 堆叠 KV: (Batch, 3, Hidden)
        kv = torch.stack([h_sent, h_soc, h_mkt], dim=1)
        
        # 计算动态全局上下文 (均值池化)
        # 注意：训练时由于 mask 的存在，这里需要做有效长度除法
        if self.training:
            valid_counts = 2.0 + mask # sent和soc始终有效为2，mkt可能为0或1
            context_state = (h_sent + h_soc + h_mkt) / valid_counts
        else:
            context_state = (h_sent + h_soc + h_mkt) / 3.0
            
        # 生成两个不同的上帝视角 Query
        q_actor = self.query_gen_actor(context_state).unsqueeze(1)  # (Batch, 1, Hidden)
        q_critic = self.query_gen_critic(context_state).unsqueeze(1) # (Batch, 1, Hidden)
        
        # 拼接 Query 并一次性进行 Attention 交互 -> (Batch, 2, Hidden)
        q_combined = torch.cat([q_actor, q_critic], dim=1) 
        
        attn_output, _ = self.attn_layer(query=q_combined, key=kv, value=kv)
        
        if torch.isnan(attn_output).any():
            attn_output = torch.nan_to_num(attn_output)

        # 拆分出专注短期方向的特征和专注长期风险的特征
        feat_actor = attn_output[:, 0, :]
        feat_critic = attn_output[:, 1, :]
        
        # 将 上下文、Actor特供、Critic特供 以及持仓状态 一起丢给下游网络
        # PPO 算法内部的 net_arch 会自动去使用这些丰富的特征
        return torch.cat([context_state, feat_actor, feat_critic, pos_state], dim=1)
    

    
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ============================================================================
# 【Baseline 1】：Vanilla MLP Fusion (早期暴力融合)
# 学术证明：证明如果不区分模态、没有注意力机制，多模态特征会互相干扰变成噪音。
# ============================================================================
class VanillaMLPFusion(BaseFeaturesExtractor):
    def __init__(self, observation_space, sent_dim=3, social_dim=4, market_dim=5, hidden_dim=64, dropout=0.2):
        # 提取后输出给 PPO 的维度
        super(VanillaMLPFusion, self).__init__(observation_space, features_dim=hidden_dim + 1)
        
        input_dim = sent_dim + social_dim + market_dim
        
        # 不分模态，把除了持仓状态外的所有特征直接拼接后扔进全连接层 (Early Fusion)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_dim),
            nn.GELU()
        )
        
    def forward(self, observations):
        if torch.isnan(observations).any():
             observations = torch.nan_to_num(observations)
             
        # 分离特征和持仓状态
        features = observations[:, :-1]
        pos_state = observations[:, -1:]
        
        feat_out = self.net(features)
        
        return torch.cat([feat_out, pos_state], dim=1)


# ============================================================================
# 【Baseline 2】：Late Fusion MLP (晚期独立融合)
# 学术证明：证明即使给每个模态独立的网络，缺乏交叉注意力(Cross-Attention)的全局统筹依然不行。
# ============================================================================
class LateFusionMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space, sent_dim=3, social_dim=4, market_dim=5, hidden_dim=64, dropout=0.2):
        super(LateFusionMLP, self).__init__(observation_space, features_dim=hidden_dim + 1)
        
        self.sent_dim, self.social_dim, self.market_dim = sent_dim, social_dim, market_dim
        
        # 各模态独立编码 (类似于你们网络的第一步)
        self.sent_encoder = nn.Sequential(nn.Linear(sent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.social_encoder = nn.Sequential(nn.Linear(social_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.market_encoder = nn.Sequential(nn.Linear(market_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        
        # 晚期线性拼接融合 (取代了 Attention)
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, observations):
        if torch.isnan(observations).any():
             observations = torch.nan_to_num(observations)
             
        idx_social = self.sent_dim
        idx_market = self.sent_dim + self.social_dim
        
        sent = observations[:, 0 : idx_social]
        social = observations[:, idx_social : idx_market]
        market = observations[:, idx_market : -1]
        pos_state = observations[:, -1:] 
        
        h_sent = self.sent_encoder(sent)
        h_soc = self.social_encoder(social)
        h_mkt = self.market_encoder(market)
        
        # 直接按维度拼接而不是作为序列
        concat_feat = torch.cat([h_sent, h_soc, h_mkt], dim=-1)
        fused_feat = self.fusion_layer(concat_feat)
        
        return torch.cat([fused_feat, pos_state], dim=1)


# ============================================================================
# 【Ablation 1】：Single-Query Fusion (单通道消融)
# 学术证明：证明 Actor (选方向) 和 Critic (判风险) 共用一个 Query 会导致任务冲突，解耦是必要的。
# ============================================================================
class SingleQueryRLFusion(BaseFeaturesExtractor):
    def __init__(self, observation_space, sent_dim=3, social_dim=4, market_dim=5, hidden_dim=64, dropout=0.2):
        # 只有 context(64) + single_feat(64) + pos(1)
        super(SingleQueryRLFusion, self).__init__(observation_space, features_dim=(hidden_dim * 2) + 1)
        
        self.sent_dim, self.social_dim, self.market_dim = sent_dim, social_dim, market_dim
        
        self.sent_encoder = nn.Sequential(nn.Linear(sent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.social_encoder = nn.Sequential(nn.Linear(social_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.market_encoder = nn.Sequential(nn.Linear(market_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        
        # 只有一个全局 Query
        self.query_gen = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attn_layer = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, batch_first=True)
        
    def forward(self, observations):
        if torch.isnan(observations).any():
             observations = torch.nan_to_num(observations)
             
        sent = observations[:, 0 : self.sent_dim]
        social = observations[:, self.sent_dim : self.sent_dim+self.social_dim]
        market = observations[:, self.sent_dim+self.social_dim : -1]
        pos_state = observations[:, -1:] 
        
        h_sent = self.sent_encoder(sent)
        h_soc = self.social_encoder(social)
        h_mkt = self.market_encoder(market)
        
        # 保留 30% mask 确保公正比对
        if self.training:
            mask = (torch.rand(h_mkt.size(0), 1, device=h_mkt.device) > 0.30).float()
            h_mkt = h_mkt * mask
            valid_counts = 2.0 + mask
            context_state = (h_sent + h_soc + h_mkt) / valid_counts
        else:
            context_state = (h_sent + h_soc + h_mkt) / 3.0
            
        kv = torch.stack([h_sent, h_soc, h_mkt], dim=1)
        q = self.query_gen(context_state).unsqueeze(1)
        
        attn_output, _ = self.attn_layer(query=q, key=kv, value=kv)
        
        if torch.isnan(attn_output).any():
            attn_output = torch.nan_to_num(attn_output)
            
        feat = attn_output.squeeze(1)
        return torch.cat([context_state, feat, pos_state], dim=1)


# ============================================================================
# 【Ablation 2】：No-Mask Dual-Query Fusion (无模态掩码消融)
# 学术证明：证明金融量化极易对市场特征产生“模态懒惰”，强制切断市场特征能逼迫网络挖掘文本潜力。
# ============================================================================
class NoMaskDualQueryRLFusion(BaseFeaturesExtractor):
    def __init__(self, observation_space, sent_dim=3, social_dim=4, market_dim=5, hidden_dim=64, dropout=0.2):
        features_dim = (hidden_dim * 3) + 1 
        super(NoMaskDualQueryRLFusion, self).__init__(observation_space, features_dim=features_dim)
        
        self.sent_dim, self.social_dim, self.market_dim = sent_dim, social_dim, market_dim
        
        self.sent_encoder = nn.Sequential(nn.Linear(sent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.social_encoder = nn.Sequential(nn.Linear(social_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.market_encoder = nn.Sequential(nn.Linear(market_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        
        self.query_gen_actor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.query_gen_critic = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attn_layer = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, batch_first=True)
        
    def forward(self, observations):
        if torch.isnan(observations).any():
             observations = torch.nan_to_num(observations)
             
        sent = observations[:, 0 : self.sent_dim]
        social = observations[:, self.sent_dim : self.sent_dim+self.social_dim]
        market = observations[:, self.sent_dim+self.social_dim : -1]
        pos_state = observations[:, -1:] 
        
        h_sent = self.sent_encoder(sent)
        h_soc = self.social_encoder(social)
        h_mkt = self.market_encoder(market)
        
        # 【关键差异】：完全去掉了 30% 的 Mask 机制，让模型 100% 看到市场数据
        context_state = (h_sent + h_soc + h_mkt) / 3.0
        kv = torch.stack([h_sent, h_soc, h_mkt], dim=1)
        
        q_actor = self.query_gen_actor(context_state).unsqueeze(1) 
        q_critic = self.query_gen_critic(context_state).unsqueeze(1) 
        
        q_combined = torch.cat([q_actor, q_critic], dim=1) 
        attn_output, _ = self.attn_layer(query=q_combined, key=kv, value=kv)
        
        if torch.isnan(attn_output).any():
            attn_output = torch.nan_to_num(attn_output)

        feat_actor = attn_output[:, 0, :]
        feat_critic = attn_output[:, 1, :]
        
        return torch.cat([context_state, feat_actor, feat_critic, pos_state], dim=1)
    

# ============================================================================
# 【消融实验 1】：No-Text (屏蔽文本情感特征)
# 目的：证明 Sentiment 对预测极端行情的领先作用
# ============================================================================
class NoTextDualQueryRLFusion(DualQueryRLFusion):
    def forward(self, observations):
        # 提取原始特征
        idx_sent = self.sent_dim
        idx_social = idx_sent + self.social_dim
        idx_market = idx_social + self.market_dim
        
        sent = observations[:, 0 : idx_sent]
        social = observations[:, idx_sent : idx_social]
        market = observations[:, idx_social : idx_market]
        pos_state = observations[:, idx_market:] 

        h_sent = self.sent_encoder(sent) * 0  # 【核心操作】强制清零
        h_soc = self.social_encoder(social)
        h_mkt = self.market_encoder(market)
        
        # 剩下的逻辑与 DualQueryRLFusion 完全一致（复用父类逻辑，建议把父类 forward 重构或拷贝）
        kv = torch.stack([h_sent, h_soc, h_mkt], dim=1)
        context_state = (h_sent + h_soc + h_mkt) / 3.0
        q_actor = self.query_gen_actor(context_state).unsqueeze(1)
        q_critic = self.query_gen_critic(context_state).unsqueeze(1)
        q_combined = torch.cat([q_actor, q_critic], dim=1) 
        attn_output, _ = self.attn_layer(query=q_combined, key=kv, value=kv)
        return torch.cat([context_state, attn_output[:, 0, :], attn_output[:, 1, :], pos_state], dim=1)

# ============================================================================
# 【消融实验 2】：No-Social (屏蔽社交热度特征)
# 目的：证明 Retweets/Likes 对市场流动性的贡献
# ============================================================================
class NoSocialDualQueryRLFusion(DualQueryRLFusion):
    def forward(self, observations):
        idx_sent = self.sent_dim
        idx_social = idx_sent + self.social_dim
        idx_market = idx_social + self.market_dim
        
        sent = observations[:, 0 : idx_sent]
        social = observations[:, idx_sent : idx_social]
        market = observations[:, idx_social : idx_market]
        pos_state = observations[:, idx_market:] 

        h_sent = self.sent_encoder(sent)
        h_soc = self.social_encoder(social) * 0 # 【核心操作】强制清零
        h_mkt = self.market_encoder(market)
        
        kv = torch.stack([h_sent, h_soc, h_mkt], dim=1)
        context_state = (h_sent + h_soc + h_mkt) / 3.0
        q_actor = self.query_gen_actor(context_state).unsqueeze(1)
        q_critic = self.query_gen_critic(context_state).unsqueeze(1)
        q_combined = torch.cat([q_actor, q_critic], dim=1) 
        attn_output, _ = self.attn_layer(query=q_combined, key=kv, value=kv)
        return torch.cat([context_state, attn_output[:, 0, :], attn_output[:, 1, :], pos_state], dim=1)

# ============================================================================
# 【最新 SOTA 对比模型】：Gated Cross-Attention Fusion (复刻自2024顶会论文逻辑)
# 学术证明：作为当前最强基准。模拟“市场查询文本 -> 联合查询社交”的级联门控交叉注意力。
# ============================================================================
class GatedCrossAttentionFusion(BaseFeaturesExtractor):
    def __init__(self, observation_space, sent_dim=3, social_dim=4, market_dim=5, hidden_dim=64, dropout=0.2):
        # 最终输出的特征维度 = hidden_dim + pos_state(1)
        super(GatedCrossAttentionFusion, self).__init__(observation_space, features_dim=hidden_dim + 1)
        
        self.sent_dim = sent_dim
        self.social_dim = social_dim
        self.market_dim = market_dim
        
        # 1. 独立编码器 (对应原论文的 IndicatorEncoder, DocEncoder 等)
        self.market_encoder = nn.Sequential(nn.Linear(market_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.sent_encoder = nn.Sequential(nn.Linear(sent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.social_encoder = nn.Sequential(nn.Linear(social_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        
        # 2. 级联交叉注意力层 (对应原论文的 cross_att_encoder1 和 2)
        # 很多顶会代码用自己写的 Attention，但在 PyTorch 中直接用 nn.MultiheadAttention 效率更高且数学等价
        self.cross_att_1 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, batch_first=True)
        self.cross_att_2 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, batch_first=True)
        
        # 3. 门控融合单元 (对应原论文 KnowFusionModel 里的 GLU 机制)
        # GLU 会将输入维度减半，因此输入设为 hidden_dim * 2，输出恰好是 hidden_dim
        self.gated_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GLU(dim=-1),
            nn.Dropout(dropout)
        )

    def forward(self, observations):
        if torch.isnan(observations).any():
             observations = torch.nan_to_num(observations)
             
        # 切片提取特征
        sent = observations[:, 0 : self.sent_dim]
        social = observations[:, self.sent_dim : self.sent_dim+self.social_dim]
        market = observations[:, self.sent_dim+self.social_dim : -1]
        pos_state = observations[:, -1:] 
        
        # 映射到隐向量空间
        h_mkt = self.market_encoder(market).unsqueeze(1)  # (Batch, 1, Hidden) 作为 Query
        h_sent = self.sent_encoder(sent).unsqueeze(1)     # (Batch, 1, Hidden) 作为 Key/Value
        h_soc = self.social_encoder(social).unsqueeze(1)  # (Batch, 1, Hidden) 作为 Key/Value
        
        # --- 步骤 1：市场 Query 文本 (Cross Attention 1) ---
        # 逻辑：在当前的市场行情下，文本舆情中哪些情绪对我们最重要？
        cross_emb_1, _ = self.cross_att_1(query=h_mkt, key=h_sent, value=h_sent)
        
        # --- 步骤 2：市场+文本联合 Query 社交热度 (Cross Attention 2) ---
        # 逻辑：结合了当前行情与情绪后，再去社交热度里确认该情绪是否有足够的影响力穿透
        cross_emb_2, _ = self.cross_att_2(query=cross_emb_1, key=h_soc, value=h_soc)
        
        # 降维处理
        cross_emb_2 = cross_emb_2.squeeze(1) # (Batch, Hidden)
        h_mkt_sq = h_mkt.squeeze(1)          # (Batch, Hidden)
        
        # --- 步骤 3：门控融合 (Gated Fusion) ---
        # 将原始的市场特征与交叉注意力提取到的多模态特征拼接，通过 GLU 门控决定多少舆情可以流入决策
        concat_feat = torch.cat([h_mkt_sq, cross_emb_2], dim=-1) # (Batch, Hidden * 2)
        final_feat = self.gated_fusion(concat_feat)              # (Batch, Hidden)
        
        # 拼接持仓状态并输出给 RL PPO 的主干网络
        return torch.cat([final_feat, pos_state], dim=1)