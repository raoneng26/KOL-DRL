from networkx.readwrite import json_graph
import networkx as nx
import json 
import matplotlib.pyplot as plt
from test_model import TestModel
import argparse
from baichuan import BaichuanLLM
from agent_mesa import TotLog


def get_event_log():
    print('event log')

    cur_log = TotLog.get_event_log()
    with open('log/cur_log.json', 'w', encoding='utf-8') as f:
        json.dump(cur_log, f)
        

# API_KEY = 'sk-f900bdfa95dba3143546b63b73768e0e'
API_KEY = 'sk-36a13fc3f2bfe35784a438ef780d6b4e' 




parser = argparse.ArgumentParser(description='run the Mesa model.')

# 添加配置文件作为参数
parser.add_argument('filename', metavar='config_file', type=str,
                   help='The config file for the sim')
parser.add_argument('round', metavar='round_num', type=int,
                   help='The round number of the sim')


args = parser.parse_args()

tar_file = args.filename

with open(tar_file, 'r', encoding='utf-8') as f:
    config_file = json.load(f) 

#获取配置
G = json_graph.node_link_graph(config_file['graph'])
person_list = config_file['person']

#绘制社交图
nx.draw(G, with_labels=True)
plt.savefig('graph.png')


llm = BaichuanLLM(API_KEY, 5)

#引入模型
model = TestModel(G, person_list, llm)

TotLog.init_log(len(person_list), if_event=True)

for i in range(args.round):
    model.step()

#输出Log

TotLog.write_log('./log/')
get_event_log()

# python run.py case_lite2.json  5