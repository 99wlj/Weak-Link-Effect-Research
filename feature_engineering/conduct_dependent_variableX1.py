import pandas as pd
import datetime
import os
import networkx as nx
from node2vec import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity

# 输入数据
base_dir = r"F:\WLJ\Weak-Link-Effect-Research"
data_dir = os.path.join(base_dir, "data")
csv_file = os.path.join(data_dir, "Keyword_LinkStrength_ByWindow_20250903_212630.csv")

df = pd.read_csv(csv_file)

all_edges_features = []

# 遍历每个时间窗口
windows = df[['window_start','window_end']].drop_duplicates().values.tolist()

for idx, (w_start, w_end) in enumerate(windows, 1):
    print(f"\n==== 处理第 {idx}/{len(windows)} 个窗口: {w_start}-{w_end} ====")

    df_window = df[(df['window_start']==w_start) & (df['window_end']==w_end)]

    # 用已有边数据构建网络
    G = nx.Graph()
    for _, row in df_window.iterrows():
        u, v = row['node_u'], row['node_v']
        w = row['link_strength']
        G.add_edge(u, v, weight=w)

    print(f"构建网络完成，节点数: {G.number_of_nodes()}，边数: {G.number_of_edges()}")

    # 计算节点度（为了计算度差）
    degrees = dict(G.degree(weight='weight'))

    # 计算剩下三个特征
    # 2. 度差
    edge_deg_diff = { (u,v): abs(degrees[u] - degrees[v]) for u,v in G.edges() }

    # 3. 边介数中心性
    edge_betweenness = nx.edge_betweenness_centrality(G, weight='weight')

    # 技术距离完善了三种方法，择优
    # 4. 技术距离 (Node2Vec)

    print("开始 Node2Vec 训练...")
    node2vec = Node2Vec(G, dimensions=64, walk_length=10, num_walks=50, workers=1, seed=42)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    embeddings = {node: model.wv.get_vector(node) for node in G.nodes()}

    edge_tech_dist = {}
    for u, v in G.edges():
        sim = cosine_similarity([embeddings[u]], [embeddings[v]])[0][0]
        dist = 1 - sim
        edge_tech_dist[(u,v)] = dist

    # 4替换方法一
    # 4. 技术距离 (Jaccard 相似度)
    print("开始计算 Jaccard 技术距离...")
    edge_tech_dist = {}
    for u, v in G.edges():
        neighbors_u = set(G.neighbors(u))
        neighbors_v = set(G.neighbors(v))
        union_size = len(neighbors_u | neighbors_v)
        if union_size == 0:
            sim = 0
        else:
            sim = len(neighbors_u & neighbors_v) / union_size
        edge_tech_dist[(u, v)] = 1 - sim  # 技术距离 = 1 - Jaccard相似度
    print("Jaccard 技术距离计算完成")

    # 4替换方法二
    # 4. 技术距离 (最短路径)
    print("开始计算最短路径技术距离...")
    edge_tech_dist = {}
    lengths = dict(nx.all_pairs_shortest_path_length(G, weight="weight"))
    max_dist = max(max(d.values()) for d in lengths.values())  # 归一化用

    for u, v in G.edges():
        dist = lengths[u].get(v, max_dist)  # 如果不连通，取最大值
        edge_tech_dist[(u, v)] = dist / max_dist  # 归一化到 [0,1]
    print("最短路径技术距离计算完成")



    # 保存结果
    for u, v, data in G.edges(data=True):
        row = df_window[((df_window['node_u']==u) & (df_window['node_v']==v)) | 
                        ((df_window['node_u']==v) & (df_window['node_v']==u))]
        strength_type = row['strength_type'].values[0]

        all_edges_features.append({
            'window_start': w_start,
            'window_end': w_end,
            'node_u': u,
            'node_v': v,
            'link_strength': row['link_strength'].values[0],  # 直接用原
            'degree_difference': edge_deg_diff[(u,v)],
            'betweenness': edge_betweenness[(u,v)],
            'tech_distance': edge_tech_dist[(u,v)],
        })

    print(f"窗口 {w_start}-{w_end} 处理完成，结果已记录。")

# 输出为DataFrame
edges_df = pd.DataFrame(all_edges_features)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(data_dir, f"Keyword_Network_Features_By_Window_{timestamp}.csv")
edges_df.to_csv(output_file, index=False)
print(f"\n输出完成：{output_file}")
