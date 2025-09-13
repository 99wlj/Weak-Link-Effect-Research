"""
基本特征X1计算脚本（修正版）
基于平衡数据集的样本计算四个基本特征：链接强度、度差、边介数中心性、技术距离
以df_samples为主，只计算样本中涉及的边的特征
"""

import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# 配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 输入文件
BALANCED_DATASET = os.path.join(DATA_DIR, "RES01_balanced_dataset_1_1_20250913_225013.csv")
LINK_STRENGTH_FILE = os.path.join(DATA_DIR, "Keyword_LinkStrength_ByWindow_20250903_212630.csv")

print("=" * 80)
print("基本特征X1计算（以样本为主）")
print("=" * 80)

# 1. 读取数据
print("\n[1] 读取数据...")
df_samples = pd.read_csv(BALANCED_DATASET)
df_links_all = pd.read_csv(LINK_STRENGTH_FILE)

print(f"  平衡样本数: {len(df_samples):,}")
print(f"  全部链接数: {len(df_links_all):,}")

# 2. 提取样本中涉及的所有边对和时间窗口
print("\n[2] 提取样本涉及的边...")

# 获取样本中所有唯一的(窗口, 节点对)组合
sample_edges = set()
for _, row in df_samples.iterrows():
    # 注意：样本使用的是window_t作为当前时间窗口
    key1 = (row['window_t_start'], row['window_t_end'], row['node_u'], row['node_v'])
    key2 = (row['window_t_start'], row['window_t_end'], row['node_v'], row['node_u'])  # 反向
    sample_edges.add(key1)
    sample_edges.add(key2)

print(f"  样本涉及的唯一边对数: {len(sample_edges):,}")

# 过滤df_links，只保留样本中涉及的边
df_links_filtered = []
for _, row in df_links_all.iterrows():
    key = (row['window_start'], row['window_end'], row['node_u'], row['node_v'])
    if key in sample_edges:
        df_links_filtered.append(row)

df_links = pd.DataFrame(df_links_filtered)
print(f"  过滤后链接数: {len(df_links):,}")

# 3. 获取样本涉及的时间窗口
print("\n[3] 处理时间窗口...")
sample_windows = df_samples[['window_t_start', 'window_t_end']].drop_duplicates()
print(f"  样本涉及的时间窗口数: {len(sample_windows)}")

# 4. 初始化特征存储
df_samples['link_strength'] = np.nan
df_samples['degree_difference'] = np.nan
df_samples['betweenness'] = np.nan
df_samples['tech_distance'] = np.nan

# 5. 按时间窗口计算特征
print("\n[4] 计算网络特征...")

for idx, (w_start, w_end) in enumerate(sample_windows.values, 1):
    print(f"\n  处理窗口 {idx}/{len(sample_windows)}: {w_start}-{w_end}")
    
    # 获取该窗口的样本
    window_sample_mask = (df_samples['window_t_start'] == w_start) & \
                         (df_samples['window_t_end'] == w_end)
    window_samples = df_samples[window_sample_mask]
    print(f"    该窗口样本数: {len(window_samples):,}")
    
    # 获取该窗口的所有链接（包括样本中的边和其他相关边用于构建完整网络）
    df_window_links = df_links[(df_links['window_start'] == w_start) & 
                                (df_links['window_end'] == w_end)]
    
    if len(df_window_links) == 0:
        print(f"    警告：窗口 {w_start}-{w_end} 没有链接数据")
        continue
    
    # 构建网络（使用所有相关边以获得准确的网络结构特征）
    G = nx.Graph()
    
    # 添加所有边
    for _, row in df_window_links.iterrows():
        G.add_edge(row['node_u'], row['node_v'], weight=row['link_strength'])
    
    # 确保样本中的所有节点都在网络中（即使是孤立节点）
    sample_nodes = set(window_samples['node_u'].unique()) | \
                   set(window_samples['node_v'].unique())
    for node in sample_nodes:
        if node not in G:
            G.add_node(node)
    
    print(f"    网络规模: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    
    # === 批量计算特征 ===
    
    # 特征1: 链接强度
    # 创建双向查找字典
    link_strength_dict = {}
    for _, row in df_window_links.iterrows():
        link_strength_dict[(row['node_u'], row['node_v'])] = row['link_strength']
        link_strength_dict[(row['node_v'], row['node_u'])] = row['link_strength']
    
    # 特征2: 度差
    degrees = dict(G.degree(weight='weight'))
    
    # 特征3: 边介数中心性
    print(f"    计算边介数中心性...")
    if G.number_of_edges() > 0:
        edge_betweenness = nx.edge_betweenness_centrality(G, weight='weight', normalized=True)
        # 创建双向查找
        edge_betweenness_dict = {}
        for (u, v), val in edge_betweenness.items():
            edge_betweenness_dict[(u, v)] = val
            edge_betweenness_dict[(v, u)] = val
    else:
        edge_betweenness_dict = {}
    
    # 特征4: 技术距离（Jaccard）
    print(f"    计算技术距离...")
    neighbors = {node: set(G.neighbors(node)) for node in G.nodes()}
    
    def compute_jaccard_distance(u, v):
        if u not in neighbors or v not in neighbors:
            return 1.0
        neighbors_u = neighbors[u]
        neighbors_v = neighbors[v]
        
        # 处理特殊情况
        if len(neighbors_u) == 0 and len(neighbors_v) == 0:
            return 1.0  # 两个孤立节点
        
        union_size = len(neighbors_u | neighbors_v)
        if union_size == 0:
            return 1.0
        
        jaccard_sim = len(neighbors_u & neighbors_v) / union_size
        return 1 - jaccard_sim
    
    # 为该窗口的每个样本计算特征
    for idx_sample in window_samples.index:
        u = df_samples.loc[idx_sample, 'node_u']
        v = df_samples.loc[idx_sample, 'node_v']
        
        # 链接强度
        link_str = link_strength_dict.get((u, v), 0)  # 如果不存在则为0
        df_samples.loc[idx_sample, 'link_strength'] = link_str
        
        # 度差
        deg_u = degrees.get(u, 0)
        deg_v = degrees.get(v, 0)
        df_samples.loc[idx_sample, 'degree_difference'] = abs(deg_u - deg_v)
        
        # 边介数中心性
        betw = edge_betweenness_dict.get((u, v), 0)
        df_samples.loc[idx_sample, 'betweenness'] = betw
        
        # 技术距离
        tech_dist = compute_jaccard_distance(u, v)
        df_samples.loc[idx_sample, 'tech_distance'] = tech_dist
    
    print(f"    窗口 {w_start}-{w_end} 特征计算完成")

# 6. 特征后处理
print("\n[5] 特征后处理...")

# 检查缺失值
feature_cols = ['link_strength', 'degree_difference', 'betweenness', 'tech_distance']
missing_counts = df_samples[feature_cols].isna().sum()

if missing_counts.sum() > 0:
    print("  缺失值统计:")
    for col, count in missing_counts.items():
        if count > 0:
            print(f"    {col}: {count} 个缺失值")
    
    # 填充缺失值
    print("  填充缺失值...")
    df_samples['link_strength'] = df_samples['link_strength'].fillna(0)
    df_samples['degree_difference'] = df_samples['degree_difference'].fillna(0)
    df_samples['betweenness'] = df_samples['betweenness'].fillna(0)
    df_samples['tech_distance'] = df_samples['tech_distance'].fillna(1)

# 7. 特征标准化
print("\n[6] 特征标准化...")

from sklearn.preprocessing import MinMaxScaler

# 保存原始值
for col in feature_cols:
    df_samples[f'{col}_raw'] = df_samples[col]

# MinMax标准化到[0,1]
scaler = MinMaxScaler()

# 处理特殊情况：如果某个特征的所有值都相同
for col in feature_cols:
    if df_samples[col].std() == 0:
        print(f"  警告: {col} 的所有值相同，跳过标准化")
        df_samples[f'{col}_scaled'] = df_samples[col]
    else:
        df_samples[f'{col}_scaled'] = scaler.fit_transform(df_samples[[col]])

# 用标准化后的值替换原始特征列
for col in feature_cols:
    df_samples[col] = df_samples[f'{col}_scaled']
    df_samples.drop(f'{col}_scaled', axis=1, inplace=True)

print("  特征已标准化到 [0, 1] 区间")

# 8. 最终数据整理
print("\n[7] 整理最终数据...")

# 确定输出列顺序
output_cols = [
    'sample_id', 
    'window_t_start', 'window_t_end', 
    'window_t1_start', 'window_t1_end',
    'node_u', 'node_v', 
    'y',
    'link_strength', 'degree_difference', 'betweenness', 'tech_distance',
    'link_strength_raw', 'degree_difference_raw', 'betweenness_raw', 'tech_distance_raw',
    'window_label'
]

# 只保留存在的列
available_cols = [col for col in output_cols if col in df_samples.columns]
df_final = df_samples[available_cols].copy()

# 9. 保存结果
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"RES02_features_X1_{timestamp}.csv"
output_path = os.path.join(DATA_DIR, output_filename)

df_final.to_csv(output_path, index=False, encoding='utf-8-sig')

print("\n" + "=" * 80)
print(f"✅ 特征X1计算完成！")
print(f"   输出文件: {output_filename}")
print(f"   路径: {output_path}")
print(f"   样本数: {len(df_final):,}")
print("=" * 80)

# 10. 生成统计报告
print("\n[8] 生成统计报告...")

report_filename = f"RES02_features_X1_report_{timestamp}.txt"
report_path = os.path.join(DATA_DIR, report_filename)

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("基本特征X1计算报告\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("数据概况:\n")
    f.write(f"  样本总数: {len(df_final):,}\n")
    f.write(f"  正样本数: {(df_final['y']==1).sum():,}\n")
    f.write(f"  负样本数: {(df_final['y']==0).sum():,}\n\n")
    
    f.write("特征统计（标准化后）:\n")
    stats = df_final[feature_cols].describe()
    f.write(stats.to_string())
    f.write("\n\n")
    
    f.write("特征统计（原始值）:\n")
    raw_cols = [f'{col}_raw' for col in feature_cols]
    stats_raw = df_final[raw_cols].describe()
    f.write(stats_raw.to_string())
    f.write("\n\n")
    
    f.write("按Y值分组的特征均值（原始值）:\n")
    grouped_stats = df_final.groupby('y')[raw_cols].mean()
    f.write(grouped_stats.to_string())
    f.write("\n\n")
    
    f.write("特征相关性矩阵:\n")
    corr_matrix = df_final[feature_cols].corr()
    f.write(corr_matrix.to_string())
    f.write("\n\n")
    
    f.write("特征说明:\n")
    f.write("  1. link_strength: 链接强度（共现频率归一化）\n")
    f.write("  2. degree_difference: 度差（节点度数差的绝对值）\n")
    f.write("  3. betweenness: 边介数中心性\n")
    f.write("  4. tech_distance: 技术距离（基于Jaccard相似度）\n")

print(f"📊 统计报告已生成: {report_filename}")

# 11. 数据验证
print("\n[9] 数据验证...")
print(f"  特征完整性: {(~df_final[feature_cols].isna().any(axis=1)).sum()}/{len(df_final)} 个样本特征完整")
print(f"  Y值分布: Y=1: {(df_final['y']==1).sum()}, Y=0: {(df_final['y']==0).sum()}")

# 简单的特征差异分析
print("\n  Y=0 vs Y=1 特征均值差异（原始值）:")
for col in raw_cols:
    mean_0 = df_final[df_final['y']==0][col].mean()
    mean_1 = df_final[df_final['y']==1][col].mean()
    diff_pct = (mean_1 - mean_0) / mean_0 * 100 if mean_0 != 0 else 0
    print(f"    {col}: Y=0:{mean_0:.4f}, Y=1:{mean_1:.4f}, 差异:{diff_pct:+.1f}%")

print("\n✅ 所有任务完成！")
