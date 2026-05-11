import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import json
from matplotlib.cm import get_cmap

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义智能体类型及其数量
agent_types = {
    '普通公众': 440,
    '患者及家属': 140,
    '医学生': 100,
    '政策制定者': 40,
    '医学专家': 100,
    '教育工作者': 60,
    '媒体': 80,
    '协和校友': 40
}

# 创建一个无向图
G = nx.Graph()

# 节点总数
total_nodes = 1000

# 为每个节点分配类型
node_types = []
node_colors = []

# 颜色映射
color_map = get_cmap('tab10')
colors = {agent_type: color_map(i % 10) for i, agent_type in enumerate(agent_types.keys())}

# 创建节点并分配类型
node_id = 0
for agent_type, count in agent_types.items():
    for i in range(count):
        G.add_node(node_id, type=agent_type)
        node_types.append(agent_type)
        node_colors.append(colors[agent_type])
        node_id += 1

# 创建全连接网络（为了可视化效果，我们只添加一部分边）
# 全连接网络边数太多会导致图形过于密集，不易观察
# 这里我们采用随机抽样的方式添加一部分边

# 首先，确保每个节点至少与同类型的其他节点相连
for agent_type, count in agent_types.items():
    nodes_of_type = [n for n, data in G.nodes(data=True) if data['type'] == agent_type]
    for i in range(len(nodes_of_type)):
        for j in range(i+1, len(nodes_of_type)):
            if random.random() < 0.3:  # 30%的概率添加同类型节点间的边
                G.add_edge(nodes_of_type[i], nodes_of_type[j])

# 然后，添加不同类型节点之间的一些边
for i in range(total_nodes):
    for j in range(i+1, total_nodes):
        if G.nodes[i]['type'] != G.nodes[j]['type'] and random.random() < 0.05:  # 5%的概率添加不同类型节点间的边
            G.add_edge(i, j)

# 使用spring_layout布局算法
pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)

# 创建图形
plt.figure(figsize=(16, 12))

# 绘制节点
nx.draw_networkx_nodes(G, pos, node_size=30, node_color=node_colors, alpha=0.8)

# 绘制边
nx.draw_networkx_edges(G, pos, width=0.1, alpha=0.3)

# 创建图例
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[agent_type], 
                             markersize=10, label=f'{agent_type} (n={count})') 
                  for agent_type, count in agent_types.items()]

plt.legend(handles=legend_handles, loc='upper right', title='智能体类型')

# 设置标题
plt.title('社会关系网络图 - 全连接拓扑网络(1000节点)', fontsize=16)

# 移除坐标轴
plt.axis('off')

# 保存图形
plt.savefig('social_network_visualization.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()

print("网络图已生成并保存为 'social_network_visualization.png'")

# 输出一些网络统计信息
print(f"\n网络统计信息:")
print(f"节点总数: {G.number_of_nodes()}")
print(f"边总数: {G.number_of_edges()}")
print(f"网络密度: {nx.density(G):.4f}")

# 输出各类型节点的数量
print("\n各类型节点数量:")
for agent_type, count in agent_types.items():
    print(f"{agent_type}: {count}")