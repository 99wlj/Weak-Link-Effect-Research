import pandas as pd
import numpy as np
import ast
import re
from itertools import combinations
import networkx as nx
import datetime
import os

# 参数设置
base_dir = r"F:\WLJ\Weak-Link-Effect-Research"
data_dir = os.path.join(base_dir, "data")
csv_file = os.path.join(data_dir, "Keywords_Processed_20250903_211104.csv")

window_length = 3  # 年
step_size = 2      # 年

# 读取数据
print("读取数据中...")
df = pd.read_csv(csv_file)
df['keywords'] = df['keywords'].apply(lambda x: ast.literal_eval(x))
df['keywords'] = df['keywords'].apply(lambda kws: [kw.lower().strip() for kw in kws])
print(f"数据读取完成，共 {len(df)} 条记录。")

# 构建时间窗口
years = sorted(df['earliest_publn_year'].unique())
windows = []
start_year = min(years)
end_year = max(years)

while start_year <= end_year:
    windows.append((start_year, start_year + window_length - 1))
    start_year += step_size

print("时间窗口:", windows)

# 初始化结果
all_edges_features = []

# 遍历时间窗口
for idx, (w_start, w_end) in enumerate(windows, 1):
    print(f"\n处理第 {idx}/{len(windows)} 个窗口: {w_start}-{w_end}")
    df_window = df[(df['earliest_publn_year'] >= w_start) & (df['earliest_publn_year'] <= w_end)]
    print(f"窗口内专利数: {len(df_window)}")

    # 构建关键词共现网络
    G = nx.Graph()
    for kws in df_window['keywords']:
        for kw1, kw2 in combinations(set(kws), 2):
            if G.has_edge(kw1, kw2):
                G[kw1][kw2]['weight'] += 1
            else:
                G.add_edge(kw1, kw2, weight=1)

    print(f"网络完成，节点: {G.number_of_nodes()}，边: {G.number_of_edges()}")

    # 节点度
    degrees = dict(G.degree(weight='weight'))

    # 链路强度
    edge_wts = {}
    for u, v, data in G.edges(data=True):
        neighbors_u = set(G.neighbors(u))
        neighbors_v = set(G.neighbors(v))
        common_neighbors = neighbors_u & neighbors_v
        numerator = sum(G[u][k]['weight'] + G[v][k]['weight'] for k in common_neighbors)
        denominator = degrees[u] + degrees[v] - 2 * data['weight']
        edge_wts[(u,v)] = numerator / denominator if denominator != 0 else 0

    # 强/弱链接分类
    wts_values = list(edge_wts.values())
    median_wts = np.median(wts_values) if wts_values else 0
    edge_strength_type = { (u,v): 'strong' if edge_wts[(u,v)] >= median_wts else 'weak' for u,v in G.edges() }

    # 保存结果
    for u,v in G.edges():
        all_edges_features.append({
            'window_start': w_start,
            'window_end': w_end,
            'node_u': u,
            'node_v': v,
            'link_strength': edge_wts[(u,v)],
            'strength_type': edge_strength_type[(u,v)]
        })

    print(f"窗口 {w_start}-{w_end} 完成")

# 输出
edges_df = pd.DataFrame(all_edges_features)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(data_dir, f"Keyword_LinkStrength_ByWindow_{timestamp}.csv")
edges_df.to_csv(output_file, index=False)
print(f"\n输出完成：{output_file}")
