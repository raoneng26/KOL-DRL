from test_agent import TestAgent
from thread_send import ThreadSend

from agent_mesa import ModelBase

from agent_mesa import TotLog
import json
import time

class SocialPost:
    def __init__(self, content):
        self.content = content  # KOL推文内容
        self.likes = 0
        self.retweets = 0
        self.comments = []  # 存储格式: {"agent_id": id, "text": "...", "likes": 0, "retweets": 0, "replies": 0}

    def to_df_row(self):
        # 导出为预测模型需要的格式
        return {
            "post_content": self.content,
            "post_likes": self.likes,
            "post_retweets": self.retweets,
            "comment_data": self.comments
        }
    
#测试样例模型
# 【test_model.py】

class TestModel(ModelBase):
        #根据config生成model
    def __init__(self, tar_graph, person_list, llm):
        """
        初始化对话系统中的每个人物和他们的对话流程。

        :param tar_graph: 目标图，表示对话系统的结构。
        :param person_list: 人物列表，包含所有参与对话的人物信息。
        :param llm: 语言模型，用于生成对话内容。
        """
        
        super().__init__(tar_graph, llm)
        
        #设置Agent        
        for cur_id in range(len(person_list)):
            cur_person = person_list[cur_id]
            cur_agent = TestAgent(cur_id, self, cur_person, None)
            self.add_agent(cur_agent, cur_id)
            

    def step(self):
        current_time = self.schedule.time
        print(f"\n>>> 舆情仿真 Step: {current_time} 开始")
        
        # 加载推文
        try:
            with open('content/%d.txt' % (current_time + 1), encoding='utf-8') as f:
                kol_tweet = f.read()
        except FileNotFoundError:
            print("未找到推文，跳过。")
            return 1

        # 初始化本次舆情的数据容器
        self.current_post_data = {
            "post_content": kol_tweet,
            "post_likes": 0,
            "post_retweets": 0,
            "comments": [] 
        }

        # --- 阶段一：Agent 对 KOL 主贴做出反应 ---
        def process_post_reaction(agent):
            res = agent.react_to_post(kol_tweet)
            if res:
                if res.get('like'): self.current_post_data["post_likes"] += 1
                if res.get('retweet'): self.current_post_data["post_retweets"] += 1
                comment_text = res.get('comment')
                if comment_text and comment_text.lower() != "null":
                    # --- [重点修复位置]：必须在这里初始化所有后续需要的键 ---
                    self.current_post_data["comments"].append({
                        "agent_id": agent.unique_id,
                        "agent_role": agent.description.get('general', 'investor'),
                        "text": comment_text,
                        "likes": 0,
                        "retweets": 0,
                        "replies": 0,
                        "sub_comments": []  # <--- 必须加上这一行，否则 Phase 2 会 KeyError
                    })

        post_thread = ThreadSend(thread_num=10)
        for agent in self.agent_list:
            post_thread.add_task(process_post_reaction, (agent,))
        post_thread.start_thread()

        # --- 阶段二：Agent 对产生的评论进行二级互动 ---
        if len(self.current_post_data["comments"]) > 0:
            print(f"Phase 2: Agents reacting to {len(self.current_post_data['comments'])} comments...")
            
            def process_comment_reaction(agent, target_comment_index):
                # 获取目标一级评论
                target = self.current_post_data["comments"][target_comment_index]
                if target['agent_id'] == agent.unique_id: return
                
                # 调用我们在 test_agent.py 中增加二级文本支持的函数
                res = agent.react_to_comment(kol_tweet, target)
                if res:
                    if res.get('like'): target['likes'] += 1
                    if res.get('retweet'): target['retweets'] += 1
                    
                    # 处理二级评论文本
                    if res.get('reply') is True:
                        reply_text = res.get('reply_content')
                        if reply_text and reply_text.lower() != "null":
                            # 此时 target['sub_comments'] 已经在 Phase 1 初始化好了
                            target['sub_comments'].append({
                                "agent_id": agent.unique_id,
                                "agent_role": agent.description.get('general', 'investor'),
                                "text": reply_text,
                                "likes": 0
                            })
                            target['replies'] += 1 # 增加回复计数

            comment_thread = ThreadSend(thread_num=10)
            for agent in self.agent_list:
                import random
                # 随机互动
                sample_size = min(len(self.current_post_data["comments"]), 2)
                indices = random.sample(range(len(self.current_post_data["comments"])), sample_size)
                for idx in indices:
                    comment_thread.add_task(process_comment_reaction, (agent, idx))
            comment_thread.start_thread()

        # 保存结果
        self.save_simulation_result(current_time)
        self.schedule.time += 1
        return 0

    def save_simulation_result(self, step_num):
        file_path = f'log/market_input_step_{step_num}.json'
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.current_post_data, f, ensure_ascii=False, indent=2)