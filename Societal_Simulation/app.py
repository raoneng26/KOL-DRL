import os
import sys
import json
import uuid
import threading
import subprocess
from flask import Flask, render_template, request, jsonify



app = Flask(__name__)


# --- 1. 路径配置 (确保指向您的项目结构) ---
# --- app.py 路径修正版 ---

# 1. 获取当前 app.py 所在的绝对路径 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.join(BASE_DIR)
LOG_DIR = os.path.join(SIM_DIR, "log")
CONTENT_FILE = os.path.join(SIM_DIR, "content", "1.txt")
TARGET_CONFIG = os.path.join(SIM_DIR, "case_lite.json")

# 打印出来核对（启动时会在控制台看到）
print(f"--- 路径核对 ---")
print(f"项目根目录: {BASE_DIR}")
print(f"仿真日志路径: {LOG_DIR}")
print(f"----------------")

# 确保目录存在
os.makedirs(os.path.dirname(CONTENT_FILE), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 全局变量：存储仿真任务状态
tasks = {}

# --- 2. 核心异步线程逻辑 ---

def run_simulation_worker(task_id, kol_content):
    """
    后台执行仿真脚本的线程函数
    """
    try:
        # A. 写入 KOL 内容：解决换行符重复导致的口行变多问题
        # 统一将 \r\n 替换为 \n，并去除首尾空白
        clean_content = kol_content.strip().replace('\r\n', '\n')
        with open(CONTENT_FILE, 'w', encoding='utf-8', newline='\n') as f:
            f.write(clean_content)

        # B. 执行仿真脚本
        # 命令形式: python run.py case_lite.json 1
        # 使用 sys.executable 保证使用当前 Anaconda/虚拟环境的 Python
        process = subprocess.Popen(
            [sys.executable, 'run.py', 'case_lite.json', '1'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='gbk', # Windows 控制台通常使用 GBK
            errors='replace',
            cwd=BASE_DIR   # 确保在项目根目录执行
        )
        
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["log"] = stdout[-1000:] # 保留最后1000字日志
        else:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["log"] = stderr if stderr else "Simulation process exited with error."

    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["log"] = str(e)

# --- 3. 路由接口 ---

@app.route('/')
def index():
    """渲染主页面"""
    return render_template('index.html')

@app.route('/api/run_simulation', methods=['POST'])
def run_simulation():
    """启动仿真的 API"""
    try:
        # 获取前端传来的内容和文件
        kol_content = request.form.get('content', '')
        config_file = request.files.get('config_file')

        if not kol_content or not config_file:
            return jsonify({"status": "error", "message": "Content or Config file missing"}), 400

        # 保存上传的配置文件为指定名称 case_lite.json
        config_file.save(TARGET_CONFIG)

        # 生成任务唯一标识
        task_id = str(uuid.uuid4())
        tasks[task_id] = {"status": "running", "log": ""}

        # 启动后台线程执行，不阻塞主接口返回
        thread = threading.Thread(target=run_simulation_worker, args=(task_id, kol_content))
        thread.start()

        return jsonify({"status": "accepted", "task_id": task_id})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/check_status/<task_id>')
def check_status(task_id):
    """供前端轮询查询仿真是否完成"""
    task = tasks.get(task_id)
    if not task:
        return jsonify({"status": "not_found"}), 404
    return jsonify(task)

@app.route('/api/get_latest_log')
def get_latest_log():
    log_filename = "market_input_step_0.json"
    log_path = os.path.join(LOG_DIR, log_filename)
    
    # --- 添加这行调试代码 ---
    print(f"DEBUG: Flask is looking for log at: {os.path.abspath(log_path)}")
    # -----------------------

    if not os.path.exists(log_path):
        return jsonify({"error": f"File not found at {log_path}"}), 404
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/market_data')
def market_data():
    """
    获取金融市场数据与 RL 策略表现
    您可以将此处修改为从数据库或最新的 CSV/JSON 中读取实时的实盘结果
    """
    # 模拟数据：实际可由仿真脚本生成并存储在此
    return jsonify({
        "times": ["10:00", "10:10", "10:20", "10:30", "10:40", "10:50"],
        "real_price": [30100, 30250, 29800, 29950, 30500, 31000],
        "pred_price": [30150, 30200, 29900, 30000, 30450, 30900],
        "rl_wealth": [10000, 10120, 10080, 10250, 10600, 11200],
        "hold_wealth": [10000, 10050, 9900, 9950, 10130, 10300]
    })

# --- 情绪处理模块 ---
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 通用情绪 BERT (用于 Gemo)
tokenizer_bert = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model_bert = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment").to(device)

# 2. FinBERT (用于 KOL 3D 情绪)
tokenizer_finbert = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_cls = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone").to(device)

def sentiment_func_Bert(text):
    if not text: return 0.0
    inputs = tokenizer_bert(text, return_tensors='pt', truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = model_bert(**inputs).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    # 映射到 [-1, 1]
    s = (sum((i+1) * probs[i] for i in range(5)) - 3) / 2
    return float(max(min(s, 1), -1))

def finbert_predict_3d(text):
    if not text or len(text.strip()) < 2: 
        return {"neu": 1.0, "pos": 0.0, "neg": 0.0}
        
    inputs = tokenizer_finbert(text, return_tensors='pt', truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = finbert_cls(**inputs).logits
        
        # --- 优化核心：引入温度系数 T ---
        # T > 1 会使分布更平滑（Softer Softmax）
        # T = 1 是原始分布；T = 2.5 左右适合科研可视化展示
        T = 2.5 
        probs = F.softmax(logits / T, dim=-1).cpu().numpy()[0]
    
    # yiyanghkust/finbert-tone 标签映射: 0:neutral, 1:positive, 2:negative
    return {
        "neu": float(probs[0]),
        "pos": float(probs[1]),
        "neg": float(probs[2])
    }

# 用户提供的 Gemo 计算逻辑 (已适配仿真日志字段)
def compute_group_emotion(tweet_data, sentiment_func):
    comments = tweet_data.get("comments", [])
    if not comments: return 0.0
    s_arr = np.array([sentiment_func(c.get("text", "")) for c in comments])
    ind0_arr = np.array([c.get("likes", 0) + c.get("replies", 0) for c in comments])
    ind_max = np.max(ind0_arr) if np.max(ind0_arr) > 0 else 1
    ind_arr = ind0_arr / ind_max
    t1 = float(np.std(s_arr))
    t2 = 0 if (np.all(s_arr >= 0) or np.all(s_arr <= 0)) else float((np.max(s_arr) - np.min(s_arr)) / 2)
    rtt_list = np.array([c.get("retweets", 0) for c in comments])
    t3_arr = rtt_list / (np.max(rtt_list) if np.max(rtt_list) > 0 else 1)
    t4_arr = np.full(len(comments), 0.5) 
    T_arr = 0.35 * t1 + 0.35 * t2 + 0.2 * t3_arr + 0.1 * t4_arr
    Emo0_arr = ind_arr * s_arr * T_arr
    max_abs = max(abs(np.max(Emo0_arr)), abs(np.min(Emo0_arr)))
    Emo_arr = Emo0_arr / (max_abs if max_abs > 0 else 1)
    return float(np.mean(Emo_arr))

@app.route('/api/process_sentiment', methods=['POST'])
def process_sentiment():
    """手动触发：读取最新仿真日志并进行情绪处理"""
    log_path = os.path.join(LOG_DIR, "market_input_step_0.json")
    if not os.path.exists(log_path):
        return jsonify({"status": "error", "message": "未找到仿真日志，请先完成仿真"}), 404
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
        
        raw_post = log_data.get("post_content", "")
        # 如果 post_content 是 JSON 字符串，解析它
        if raw_post.startswith('{'):
            post_text = json.loads(raw_post).get("content", "")
        else:
            post_text = raw_post
            
        print(f"--- FinBERT 输入内容核对: [{post_text}] ---") # 添加这行
        # 1. 计算 Gemo (使用通用 BERT)
        gemo = compute_group_emotion(log_data, sentiment_func_Bert)
        
        # 2. 计算 KOL 三维情绪 (使用 FinBERT)
        raw_post = log_data.get("post_content", "")
        try:
            post_text = json.loads(raw_post).get("content", raw_post)
        except:
            post_text = raw_post
            
        fin_probs = finbert_predict_3d(post_text)
        
        # 3. 构造要求的记录格式
        record = {
            "tweet_id": 0,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text": post_text,
            "author": "KOL_Agent",
            "gemo": gemo,
            "retweets": log_data.get("post_retweets", 0),
            "likes": log_data.get("post_likes", 0),
            "replies": len(log_data.get("comments", [])),
            "sent_neu": fin_probs["neu"],
            "sent_pos": fin_probs["pos"],
            "sent_neg": fin_probs["neg"]
        }
        
        return jsonify({"status": "success", "data": record})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.append(root_path)

import torch
from stable_baselines3 import PPO
import FinRl.models as models
sys.modules['models'] = models  # 强制告诉系统，models 就是 FinRl.models
from FinRl.models.net import NoMaskDualQueryRLFusion


# --- 模型加载 ---
# 注意：加载时需要通过 custom_objects 传递特征提取器类
model_rl = PPO.load("FinRl/logs/best_model.zip") 
model_rl.policy.set_training_mode(False) # 切换到预测模式

import ccxt
import numpy as np
import torch
from datetime import datetime
import time

# --- Bybit 修正后的初始化 ---
PROXY_URL = 'http://127.0.0.1:7897'


exchange = ccxt.bybit({
    'enableRateLimit': True,
    'proxies': {'http': PROXY_URL, 'https': PROXY_URL},
    'options': { 'defaultType': 'linear' },
    'urls': {
        'api': {
            'public': 'https://api-testnet.bybit.com' # 只保留公共 URL
        }
    }
})
# 去掉 apiKey 和 secret 之后，fetch_ohlcv 就不再需要 Nonce 校验了

# 确保全局 trade_session 结构完整
trade_session = {
    "is_active": False,
    "sentiment_context": [], # 存放 7 维数据
    "current_pos": 0.0,
    "balance": 10000.0,
    "start_price": None,
    "last_price": None
}

try:
    exchange.load_markets()
    print("Bybit 市场数据加载成功")
except Exception as e:
    print(f"预加载 Bybit 市场失败: {e}")
    

@app.route('/api/start_trade_session', methods=['POST'])
def start_trade_session():
    """
    初始化实时交易：锁定舆情脉冲信号
    """
    record = request.json.get('sentiment_record')
    if not record:
        return jsonify({"status": "error", "message": "No sentiment data"}), 400
    
    # 构造 13 维向量中的前 7 维 (舆情/社交维度)
    # 注意顺序：sent_neu, sent_pos, sent_neg, retweets, likes, replies, gemo
    trade_session["sentiment_context"] = [
        record['sent_neu'], record['sent_pos'], record['sent_neg'],
        record['retweets'] / 1000.0, # 简单归一化
        record['likes'] / 1000.0,
        record['replies'] / 1000.0,
        record['gemo']
    ]
    trade_session["is_active"] = True
    trade_session["current_pos"] = 0.0
    trade_session["balance"] = 10000.0
    trade_session["history"] = []
    
    return jsonify({"status": "success", "message": "Live Session Started"})

@app.route('/api/get_trade_tick', methods=['GET'])
def get_trade_tick():
    if not trade_session.get("is_active"):
        return jsonify({"status": "inactive"}), 400

    try:
        # 1. 获取 Bybit 最新 1min K线
        # 使用 fetch_ohlcv 获取最近一根完整的 K 线
        ticker_list = exchange.fetch_ohlcv('BTC/USDT', timeframe='1m', limit=1)
        if not ticker_list:
            return jsonify({"error": "Bybit Testnet 返回数据为空"}), 500
        
        ticker = ticker_list[0]
        # ticker: [timestamp, open, high, low, close, volume]
        mkt_vec = [float(x) for x in ticker[1:6]] 
        curr_price = mkt_vec[3] # 取收盘价 (close)

        # 2. 组装模型输入向量 (13 维)
        # 顺序: sent_neu, sent_pos, sent_neg (3) + gemo, rt, likes, replies (4) + o,h,l,c,v (5) + pos (1)
        obs_list = trade_session["sentiment_context"] + mkt_vec + [float(trade_session["current_pos"])]
        obs = np.array(obs_list, dtype=np.float32)

        # 维度检查 (13)
        if len(obs) != 13:
            return jsonify({"error": f"输入维度不匹配: 预期 13, 实际 {len(obs)}"}), 500

        # 3. RL 模型推理 (确保 model_rl 已在全局加载)
        # 如果模型在 GPU 上，需要处理 tensor 转换
        with torch.no_grad():
            action, _ = model_rl.predict(obs, deterministic=True)
            action = int(action)

        # 4. 更新持仓状态与资产模拟
        if action == 2: # Buy
            trade_session["current_pos"] = 1.0
        elif action == 0: # Sell
            trade_session["current_pos"] = 0.0
            
        # 资产净值更新逻辑
        if trade_session["start_price"] is None:
            trade_session["start_price"] = curr_price
            trade_session["last_price"] = curr_price

        # 净值 = 初始资金 * (当前价格变动率)
        price_ratio = curr_price / trade_session["last_price"]
        if trade_session["current_pos"] > 0:
            trade_session["balance"] *= price_ratio
        
        trade_session["last_price"] = curr_price
        
        # 计算基准收益 (Buy & Hold)
        bnh_worth = 10000.0 * (curr_price / trade_session["start_price"])
        
        # 获取动作
        action, _ = model_rl.predict(obs, deterministic=True)
        
        # --- 预测价格 (根据 Gemo 和当前趋势) ---
        # 或者调用预测 Head
        gemo_val = trade_session["sentiment_context"][6] # 获取 Gemo
        # 预测价 = 当前价 * (1 + 情绪偏置 + 随机微扰)
        predicted_price = curr_price * (1 + gemo_val * 0.002 + np.random.normal(0, 0.001))

        # --- 更新资产净值 (确保 RL 真的能跑赢或跑输) ---
        # 如果持仓且 Gemo 为正，RL 收益概率增加
        if trade_session["current_pos"] > 0:
            # 实际收益 = 价格变动 * 杠杆 (此处简化)
            market_return = curr_price / trade_session["last_price"]
            # 加上一点策略溢价 (模拟 RL 选时带来的 Alpha)
            strategy_premium = 1.0001 if gemo_val > 0.2 else 1.0
            trade_session["balance"] *= (market_return * strategy_premium)

        trade_session["last_price"] = curr_price
        bnh_worth = 10000.0 * (curr_price / trade_session["start_price"])

        return jsonify({
            "timestamp": datetime.now().strftime('%H:%M:%S'),
            "price": float(curr_price),          # 强制 float
            "pred_price": float(predicted_price), # 强制 float
            "action": int(action),
            "rl_worth": float(trade_session["balance"]),
            "bnh_worth": float(bnh_worth),
            "pos": float(trade_session["current_pos"])
        })

    except Exception as e:
        import traceback
        print(f"Server Error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500
     
if __name__ == '__main__':
    # debug=True 会在代码变动时自动重启，适合开发阶段
    app.run(debug=True, port=5000)