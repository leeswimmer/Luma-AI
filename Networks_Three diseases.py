import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.cm as cm
import pandas as pd

# 读取CSV
file_path = "G:/Medicine LLM/figure_coding/3种疾病网络图.csv"
df = pd.read_csv(file_path)
df_clean = df.dropna()

# 构建图
G = nx.DiGraph()
for _, row in df_clean.iterrows():
    indicator = str(row.iloc[0]).strip()
    disease = str(row.iloc[1]).strip()
    G.add_node(indicator, type='indicator')
    G.add_node(disease, type='disease')
    G.add_edge(indicator, disease)

# 分类节点
indicators = [n for n, attr in G.nodes(data=True) if attr['type'] == 'indicator']
diseases = [n for n, attr in G.nodes(data=True) if attr['type'] == 'disease']

# 节点布局
radius_outer = 4.2
angle_step = 2 * np.pi / len(indicators)
indicator_pos = {
    node: (radius_outer * np.cos(i * angle_step), radius_outer * np.sin(i * angle_step))
    for i, node in enumerate(indicators)
}

radius_inner = 1.5
angle_step_disease = 2 * np.pi / len(diseases)
disease_pos = {
    node: (radius_inner * np.cos(i * angle_step_disease), radius_inner * np.sin(i * angle_step_disease))
    for i, node in enumerate(diseases)
}

pos = {**indicator_pos, **disease_pos}

colors = cm.rainbow(np.linspace(0, 1, len(indicators)))

plt.figure(figsize=(16, 16))

# 绘制外圈节点（彩虹色）
for i, node in enumerate(indicators):
    nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=[colors[i]], node_size=3500)

# 绘制内圈节点（绿色）
nx.draw_networkx_nodes(G, pos, nodelist=diseases, node_color='lightgreen', node_size=4000)

# 绘制边
nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowstyle='-|>', width=2)

# 自定义标签绘制，使文字穿过外圈节点的中心
for i, node in enumerate(indicators):
    x, y = pos[node]
    angle = np.degrees(np.arctan2(y, x))
    ha = 'center'
    va = 'center'
    alignment_angle = angle if x >= 0 else angle + 180
    plt.text(x, y, node, fontsize=35, fontname='Times New Roman',
             rotation=alignment_angle, ha=ha, va=va, weight='bold')

# 中心疾病标签
for node in diseases:
    x, y = pos[node]
    plt.text(x, y, node, fontsize=40,fontname='Times New Roman',
             ha='center', va='center', weight='bold')

# 标题
#lt.title("Network of Blood Indicators and Diseases", fontsize=30, fontname='Times New Roman')

plt.axis('off')
plt.tight_layout()
plt.show()
