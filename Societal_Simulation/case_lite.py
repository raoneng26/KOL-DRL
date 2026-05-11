import json

# 生成包含1000个节点的全连接网络结构
node_count = 10

# 生成节点
nodes = [{"id": i} for i in range(node_count)]

# 生成全连接的链接
links = []
for i in range(node_count):
    for j in range(i + 1, node_count):
        links.append({"source": i, "target": j})

# 读取person.json文件，指定编码格式为utf-8
try:
    with open('person.json', 'r', encoding='utf-8') as f:
        person = json.load(f)
except UnicodeDecodeError:
    # 如果utf-8编码也不行，尝试其他编码格式
    with open('person.json', 'r', encoding='latin1') as f:
        person = json.load(f)

# 构建图结构
graph = {
    "graph": {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": nodes,
        "links": links
    },
    "person": person
}

# 将图结构保存为新的 JSON 文件
with open('case_lite5.json', 'w', encoding='utf-8') as f:
    json.dump(graph, f, indent=4, ensure_ascii=False)