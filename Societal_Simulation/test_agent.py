from agent_mesa import AgentBase
from agent_mesa import BaseStep, JsonStep, ChoiceStep, ScoreStep
from agent_mesa import TotLog
import threading

    
#样例agent
class TestAgent(AgentBase):
    def __init__(self, unique_id, model, description, context):
        super().__init__(unique_id, model, description, context)
        
        # 针对社交媒体场景的三个Step
        # 1. 对主贴的反应 (点赞/转发/写评论文本)
        post_react_prompt = self.model.prompt_factory.get_template("post_react.txt")
        # 2. 对他人评论的反应 (点赞/转发/回复 - 仅产生计数意图)
        comment_react_prompt = self.model.prompt_factory.get_template("comment_react.txt")

        # 定义 Step
        post_step = JsonStep(0, post_react_prompt)
        comment_step = JsonStep(0, comment_react_prompt) # 注意：如果是独立链，ID通常从0开始
        
        # 注册到 chain_dict
        chain_dict = {
            'post_react': [post_step],
            'comment_react': [comment_step]
        }
        self.setup_chain(chain_dict)
        self.lock = threading.Lock() # 确保锁已初始化

    def react_to_post(self, post_content):
        """Agent 对 KOL 推文的初步反应"""
        self.lock.acquire() # 必须加锁，防止多线程同时修改同一个 Agent 的 Chain
        try:
            input_item = {
                'post_content': post_content
            }
            # 确保 Chain 清理并设置新输入
            self.chains['post_react'].set_input(input_item)
            self.chains['post_react'].run_step()
            result = self.chains['post_react'].get_output().get('json')
            return result
        except Exception as e:
            print(f"Agent {self.unique_id} react_to_post error: {e}")
            return None
        finally:
            self.lock.release() # 无论成功失败必须释放锁

    def react_to_comment(self, post_content, comment_item):
        """Agent 对他人评论的二次反应"""
        self.lock.acquire() # 必须加锁
        try:
            input_item = {
                'post_content': post_content,
                'target_agent_role': comment_item['agent_role'],
                'target_comment_text': comment_item['text']
            }
            self.chains['comment_react'].set_input(input_item)
            self.chains['comment_react'].run_step()
            result = self.chains['comment_react'].get_output().get('json')
            return result
        except Exception as e:
            print(f"Agent {self.unique_id} react_to_comment error: {e}")
            return None
        finally:
            self.lock.release()