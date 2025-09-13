"""
基本特征X1计算脚本（优化版）
基于平衡数据集计算四个基本特征：链接强度、度差、边介数中心性、技术距离
采用批量计算和向量化操作，避免循环，提高计算效率
"""

import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# 配置
DATA_DIR = 'F:/WLJ/Weak-Link-Effect-Research/data'

# 输入文件
BALANCED_DATASET = os.path.join(DATA_DIR, "RES01_balanced_dataset_1_1_20250913_225013.csv")
LINK_STRENGTH_FILE = os.path.join(DATA_DIR, "Keyword_LinkStrength_ByWindow_20250903_212630.csv")

print("=" * 80)
print("基本特征X1计算")
print("=" * 80)

# 1. 读取数据
print("\n[1] 读取数据...")
df_samples = pd.read_csv(BALANCED_DATASET)
df_links = pd.read_csv(LINK_STRENGTH_FILE)

print(f"  样本数: {len(df_samples):,}")
print(f"  链接数: {len(df_links):,}")

# 2. 预计算所有时间窗口的网络特征
print("\n[2] 批量计算网络特征...")

# 获取所有唯一的时间窗口
windows = df_links[['window_start', 'window_end']].drop_duplicates()
print(f"  时间窗口数: {len(windows)}")

# 存储所有窗口的特征
all_features = []

for idx, (w_start, w_end) in enumerate(windows.values, 1):
    print(f"\n  处理窗口 {idx}/{len(windows)}: {w_start}-{w_end}")
    
    # 获取该窗口的所有边
    df_window = df_links[(df_links['window_start'] == w_start) & 
                         (df_links['window_end'] == w_end)]
    
    # 构建网络
    G = nx.Graph()
    edge_list = df_window[['node_u', 'node_v', 'link_strength']].values
    G.add_weighted_edges_from(edge_list, weight='weight')
    
    print(f"    节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
    
    # === 批量计算特征 ===
    
    # 特征1: 链接强度（直接从df_window获取）
    link_strengths = df_window.set_index(['node_u', 'node_v'])['link_strength'].to_dict()
    
    # 特征2: 度差（向量化计算）
    degrees = dict(G.degree(weight='weight'))
    df_window['degree_u'] = df_window['node_u'].map(degrees)
    df_window['degree_v'] = df_window['node_v'].map(degrees)
    df_window['degree_difference'] = np.abs(df_window['degree_u'] - df_window['degree_v'])
    
    # 特征3: 边介数中心性（批量计算）
    print(f"    计算边介数中心性...")
    edge_betweenness = nx.edge_betweenness_centrality(G, weight='weight', normalized=True)
    # 处理双向边
    edge_betweenness_full = {}
    for (u, v), val in edge_betweenness.items():
        edge_betweenness_full[(u, v)] = val
        edge_betweenness_full[(v, u)] = val
    
    df_window['betweenness'] = df_window.apply(
        lambda row: edge_betweenness_full.get((row['node_u'], row['node_v']), 0), 
        axis=1
    )
    
    # 特征4: 技术距离（使用Jaccard，向量化计算）
    print(f"    计算技术距离...")
    
    # 预计算所有节点的邻居集合
    neighbors = {node: set(G.neighbors(node)) for node in G.nodes()}
    
    def compute_jaccard_distance(row):
        u, v = row['node_u'], row['node_v']
        if u not in neighbors or v not in neighbors:
            return 1.0  # 最大距离
        neighbors_u = neighbors[u]
        neighbors_v = neighbors[v]
        union_size = len(neighbors_u | neighbors_v)
        if union_size == 0:
            return 1.0
        jaccard_sim = len(neighbors_u & neighbors_v) / union_size
        return 1 - jaccard_sim
    
    df_window['tech_distance'] = df_window.apply(compute_jaccard_distance, axis=1)
    
    # 保留需要的特征列
    df_window_features = df_window[['window_start', 'window_end', 'node_u', 'node_v',
                                    'link_strength', 'degree_difference', 
                                    'betweenness', 'tech_distance']]
    
    all_features.append(df_window_features)
    print(f"    特征计算完成")

# 合并所有窗口的特征
print("\n[3] 合并特征数据...")
df_features = pd.concat(all_features, ignore_index=True)
print(f"  总特征记录数: {len(df_features):,}")

# 3. 将特征与样本数据关联
print("\n[4] 特征与样本关联...")

# 为了merge，需要确保窗口和节点对的匹配
# 注意：样本中的window_t对应特征中的window
df_samples_merge = df_samples.copy()
df_features_merge = df_features.copy()

# 重命名列以便merge
df_features_merge = df_features_merge.rename(columns={
    'window_start': 'window_t_start',
    'window_end': 'window_t_end'
})

# 执行merge（左连接，保留所有样本）
df_result = pd.merge(
    df_samples_merge,
    df_features_merge,
    on=['window_t_start', 'window_t_end', 'node_u', 'node_v'],
    how='left'
)

# 检查是否有未匹配的样本（可能是边方向问题）
unmatched_mask = df_result['link_strength'].isna()
if unmatched_mask.any():
    print(f"  发现 {unmatched_mask.sum()} 个未匹配样本，尝试反向匹配...")
    
    # 创建反向特征表（交换u和v）
    df_features_reverse = df_features_merge.copy()
    df_features_reverse[['node_u', 'node_v']] = df_features_reverse[['node_v', 'node_u']]
    
    # 对未匹配的样本进行反向merge
    df_unmatched = df_samples_merge[unmatched_mask].copy()
    df_unmatched_merged = pd.merge(
        df_unmatched,
        df_features_reverse,
        on=['window_t_start', 'window_t_end', 'node_u', 'node_v'],
        how='left'
    )
    
    # 更新结果
    df_result.loc[unmatched_mask] = df_unmatched_merged

# 4. 特征后处理
print("\n[5] 特征后处理...")

# 检查缺失值
missing_counts = df_result[['link_strength', 'degree_difference', 
                            'betweenness', 'tech_distance']].isna().sum()
if missing_counts.sum() > 0:
    print("  缺失值统计:")
    for col, count in missing_counts.items():
        if count > 0:
            print(f"    {col}: {count} 个缺失值")
    
    # 填充缺失值（如果边在t时刻不存在，说明是新出现的弱链接）
    df_result['link_strength'] = df_result['link_strength'].fillna(0)
    df_result['degree_difference'] = df_result['degree_difference'].fillna(0)
    df_result['betweenness'] = df_result['betweenness'].fillna(0)
    df_result['tech_distance'] = df_result['tech_distance'].fillna(1)  # 最大距离

# 5. 特征标准化（可选）
print("\n[6] 特征标准化...")

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 选择需要标准化的特征
feature_cols = ['link_strength', 'degree_difference', 'betweenness', 'tech_distance']

# 保存原始值
for col in feature_cols:
    df_result[f'{col}_raw'] = df_result[col]

# 使用MinMax标准化到[0,1]
scaler = MinMaxScaler()
df_result[feature_cols] = scaler.fit_transform(df_result[feature_cols])

print("  特征已标准化到 [0, 1] 区间")

# 6. 最终数据整理
print("\n[7] 整理最终数据...")

# 重新排序列
final_cols = [
    'sample_id', 'window_t_start', 'window_t_end', 
    'window_t1_start', 'window_t1_end',
    'node_u', 'node_v', 'y',
    'link_strength', 'degree_difference', 'betweenness', 'tech_distance',
    'link_strength_raw', 'degree_difference_raw', 'betweenness_raw', 'tech_distance_raw',
    'window_label'
]

# 确保所有列都存在
for col in final_cols:
    if col not in df_result.columns and col != 'window_label':
        print(f"  警告: 列 {col} 不存在")

df_final = df_result[[col for col in final_cols if col in df_result.columns]]

# 7. 保存结果
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

# 8. 生成特征统计报告
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

# 9. 快速验证
print("\n[9] 数据验证...")
print(f"  特征完整性: {(~df_final[feature_cols].isna().any(axis=1)).sum()}/{len(df_final)} 个样本特征完整")
print(f"  Y值分布: Y=1: {(df_final['y']==1).sum()}, Y=0: {(df_final['y']==0).sum()}")

print("\n✅ 所有任务完成！")
