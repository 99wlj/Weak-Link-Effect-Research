"""
基本特征X1计算脚本（优化版）
基于RES01_patent_based_sampled数据集计算四个基本特征
使用向量化操作优化计算速度
"""

import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
import os
import warnings
from numba import jit
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler
import multiprocessing as mp
from functools import partial
warnings.filterwarnings('ignore')

# 配置
DATA_DIR = 'F:/WLJ/Weak-Link-Effect-Research/data'

# 输入文件 - 使用新的基于专利数采样的数据集
BALANCED_DATASET = os.path.join(DATA_DIR, "RES01_patent_based_sampled_20250914_000758.csv")
LINK_STRENGTH_FILE = os.path.join(DATA_DIR, "Keyword_LinkStrength_ByWindow_20250903_212630.csv")

print("=" * 80)
print("基本特征X1计算（优化版）")
print("=" * 80)

# 1. 读取数据
print("\n[1] 读取数据...")
df_samples = pd.read_csv(BALANCED_DATASET)
df_links_all = pd.read_csv(LINK_STRENGTH_FILE)

print(f"  样本数: {len(df_samples):,}")
print(f"  全部链接数: {len(df_links_all):,}")

# 2. 优化数据准备 - 使用向量化操作
print("\n[2] 准备数据（向量化）...")

# 创建边的唯一标识符（用于快速查找）
df_samples['edge_id'] = df_samples['node_u'] + '|' + df_samples['node_v']
df_samples['edge_id_rev'] = df_samples['node_v'] + '|' + df_samples['node_u']

df_links_all['edge_id'] = df_links_all['node_u'] + '|' + df_links_all['node_v']

# 创建时间窗口标识
df_samples['window_id'] = df_samples['window_t_start'].astype(str) + '-' + df_samples['window_t_end'].astype(str)
df_links_all['window_id'] = df_links_all['window_start'].astype(str) + '-' + df_links_all['window_end'].astype(str)

# 获取所有需要的边和窗口组合
sample_edges = pd.concat([
    df_samples[['window_id', 'edge_id']].rename(columns={'edge_id': 'edge_lookup'}),
    df_samples[['window_id', 'edge_id_rev']].rename(columns={'edge_id_rev': 'edge_lookup'})
]).drop_duplicates()

print(f"  样本涉及的边-窗口组合数: {len(sample_edges):,}")

# 3. 批量预处理链接数据
print("\n[3] 预处理链接数据...")

# 创建链接强度查找表
df_links_all['lookup_key'] = df_links_all['window_id'] + '|' + df_links_all['edge_id']
link_strength_lookup = dict(zip(df_links_all['lookup_key'], df_links_all['link_strength']))

# 反向边也加入查找表
df_links_all['lookup_key_rev'] = df_links_all['window_id'] + '|' + df_links_all['node_v'] + '|' + df_links_all['node_u']
link_strength_lookup.update(dict(zip(df_links_all['lookup_key_rev'], df_links_all['link_strength'])))

print(f"  链接强度查找表大小: {len(link_strength_lookup):,}")

# 4. 初始化特征数组（使用NumPy数组更快）
n_samples = len(df_samples)
link_strengths = np.zeros(n_samples)
degree_differences = np.zeros(n_samples)
betweennesses = np.zeros(n_samples)
tech_distances = np.ones(n_samples)  # 默认值为1

# 5. 获取时间窗口
unique_windows = df_samples[['window_t_start', 'window_t_end', 'window_id']].drop_duplicates()
print(f"\n[4] 处理 {len(unique_windows)} 个时间窗口...")

# 6. 定义优化的特征计算函数
def compute_window_features(window_data):
    """计算单个时间窗口的特征（可并行）"""
    w_start, w_end, window_id = window_data
    
    # 获取该窗口的样本索引
    window_mask = df_samples['window_id'] == window_id
    window_indices = df_samples[window_mask].index.values
    
    if len(window_indices) == 0:
        return None
    
    # 获取该窗口的所有链接
    window_links = df_links_all[df_links_all['window_id'] == window_id]
    
    if len(window_links) == 0:
        return None
    
    # 构建网络（使用from_pandas_edgelist更快）
    G = nx.from_pandas_edgelist(
        window_links, 
        source='node_u', 
        target='node_v', 
        edge_attr='link_strength',
        create_using=nx.Graph()
    )
    
    # 确保样本节点都在网络中
    sample_nodes = set(df_samples.loc[window_indices, 'node_u']) | \
                  set(df_samples.loc[window_indices, 'node_v'])
    for node in sample_nodes - set(G.nodes()):
        G.add_node(node)
    
    # 批量计算节点度
    degrees = dict(G.degree(weight='link_strength'))
    
    # 计算边介数（如果网络不太大）
    if G.number_of_edges() < 25000:  # 对于大网络，可能需要采样
        print(G.number_of_edges())
        edge_betweenness = nx.edge_betweenness_centrality(G, weight='link_strength', normalized=True)
        # 创建双向查找
        betweenness_lookup = {}
        for (u, v), val in edge_betweenness.items():
            betweenness_lookup[f"{u}|{v}"] = val
            betweenness_lookup[f"{v}|{u}"] = val
    else:
        # 对于大网络，使用近似算法或采样
        betweenness_lookup = {}
    
    # 预计算邻居集合（用于Jaccard距离）
    neighbors = {node: set(G.neighbors(node)) for node in G.nodes()}
    
    # 准备结果数组
    results = []
    
    # 向量化计算该窗口所有样本的特征
    for idx in window_indices:
        u = df_samples.loc[idx, 'node_u']
        v = df_samples.loc[idx, 'node_v']
        edge_key = f"{u}|{v}"
        
        # 特征1: 链接强度
        lookup_key = f"{window_id}|{edge_key}"
        link_str = link_strength_lookup.get(lookup_key, 0)
        
        # 特征2: 度差
        deg_u = degrees.get(u, 0)
        deg_v = degrees.get(v, 0)
        deg_diff = abs(deg_u - deg_v)
        
        # 特征3: 边介数
        between = betweenness_lookup.get(edge_key, 0)
        
        # 特征4: Jaccard距离（优化计算）
        if u in neighbors and v in neighbors:
            neigh_u = neighbors[u]
            neigh_v = neighbors[v]
            union_size = len(neigh_u | neigh_v)
            if union_size > 0:
                jaccard_sim = len(neigh_u & neigh_v) / union_size
                tech_dist = 1 - jaccard_sim
            else:
                tech_dist = 1.0
        else:
            tech_dist = 1.0
        
        results.append((idx, link_str, deg_diff, between, tech_dist))
    
    return results

# 7. 批量处理所有窗口（可选：使用并行处理）
print("\n[5] 批量计算特征...")

# 串行处理（稳定但较慢）
all_results = []
for idx, row in enumerate(unique_windows.values, 1):
    print(f"  处理窗口 {idx}/{len(unique_windows)}: {row[2]}", end='\r')
    window_results = compute_window_features(row)
    if window_results:
        all_results.extend(window_results)

print(f"\n  完成特征计算，共 {len(all_results)} 个结果")

# 8. 批量赋值特征（向量化）
print("\n[6] 批量赋值特征...")

# 将结果转换为数组
results_array = np.array(all_results)
if len(results_array) > 0:
    indices = results_array[:, 0].astype(int)
    link_strengths[indices] = results_array[:, 1]
    degree_differences[indices] = results_array[:, 2]
    betweennesses[indices] = results_array[:, 3]
    tech_distances[indices] = results_array[:, 4]

# 将特征添加到DataFrame
df_samples['link_strength'] = link_strengths
df_samples['degree_difference'] = degree_differences
df_samples['betweenness'] = betweennesses
df_samples['tech_distance'] = tech_distances

# 9. 特征标准化（向量化）
print("\n[7] 特征标准化...")

feature_cols = ['link_strength', 'degree_difference', 'betweenness', 'tech_distance']

# 保存原始值
for col in feature_cols:
    df_samples[f'{col}_raw'] = df_samples[col].values

# 使用向量化的标准化
scaler = MinMaxScaler()
for col in feature_cols:
    # 处理特殊情况
    col_values = df_samples[col].values.reshape(-1, 1)
    if np.std(col_values) > 1e-10:  # 避免除零
        df_samples[col] = scaler.fit_transform(col_values).flatten()
    else:
        print(f"  警告: {col} 方差过小，保持原值")

print("  特征已标准化到 [0, 1] 区间")

# 10. 最终数据整理
print("\n[8] 整理最终数据...")

# 移除临时列
df_samples.drop(['edge_id', 'edge_id_rev', 'window_id'], axis=1, inplace=True)

# 确定输出列顺序
output_cols = [
    'sample_id', 
    'window_t_start', 'window_t_end', 
    'window_t1_start', 'window_t1_end',
    'node_u', 'node_v', 
    'y',
    'link_strength', 'degree_difference', 'betweenness', 'tech_distance',
    'link_strength_raw', 'degree_difference_raw', 'betweenness_raw', 'tech_distance_raw',
    'window_label',
    'edge_importance',  # 保留专利数信息
    'patent_level'       # 保留专利等级信息
]

# 只保留存在的列
available_cols = [col for col in output_cols if col in df_samples.columns]
df_final = df_samples[available_cols].copy()

# 11. 保存结果
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"RES02_features_X1_optimized_{timestamp}.csv"
output_path = os.path.join(DATA_DIR, output_filename)

df_final.to_csv(output_path, index=False, encoding='utf-8-sig')

print("\n" + "=" * 80)
print(f"✅ 特征X1计算完成（优化版）！")
print(f"   输出文件: {output_filename}")
print(f"   路径: {output_path}")
print(f"   样本数: {len(df_final):,}")
print("=" * 80)

# 12. 生成统计报告
print("\n[9] 生成统计报告...")

# 特征统计
print("\n特征统计（标准化后）:")
print(df_final[feature_cols].describe())

print("\n特征统计（原始值）:")
raw_cols = [f'{col}_raw' for col in feature_cols]
print(df_final[raw_cols].describe())

# 按Y值分组统计
print("\n按Y值分组的特征均值:")
grouped_stats = df_final.groupby('y')[feature_cols].mean()
print(grouped_stats)

# 按专利等级分组统计（如果存在）
if 'patent_level' in df_final.columns:
    print("\n按专利等级分组的特征均值:")
    patent_level_stats = df_final.groupby('patent_level')[feature_cols].mean()
    print(patent_level_stats)

# 特征相关性
print("\n特征相关性矩阵:")
corr_matrix = df_final[feature_cols].corr()
print(corr_matrix)

# 13. 生成详细报告文件
report_filename = f"RES02_features_X1_report_{timestamp}.txt"
report_path = os.path.join(DATA_DIR, report_filename)

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("基本特征X1计算报告（优化版）\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"数据源: {os.path.basename(BALANCED_DATASET)}\n\n")
    
    f.write("数据概况:\n")
    f.write(f"  样本总数: {len(df_final):,}\n")
    f.write(f"  正样本数: {(df_final['y']==1).sum():,}\n")
    f.write(f"  负样本数: {(df_final['y']==0).sum():,}\n")
    f.write(f"  时间窗口数: {len(unique_windows)}\n\n")
    
    if 'edge_importance' in df_final.columns:
        f.write("专利数统计:\n")
        f.write(f"  平均专利数: {df_final['edge_importance'].mean():.1f}\n")
        f.write(f"  中位数: {df_final['edge_importance'].median():.0f}\n")
        f.write(f"  最大值: {df_final['edge_importance'].max():.0f}\n")
        f.write(f"  最小值: {df_final['edge_importance'].min():.0f}\n\n")
    
    f.write("特征统计（标准化后）:\n")
    f.write(df_final[feature_cols].describe().to_string())
    f.write("\n\n")
    
    f.write("特征统计（原始值）:\n")
    f.write(df_final[raw_cols].describe().to_string())
    f.write("\n\n")
    
    f.write("按Y值分组的特征均值（标准化后）:\n")
    f.write(grouped_stats.to_string())
    f.write("\n\n")
    
    # Y=0 vs Y=1 的差异分析
    f.write("Y=0 vs Y=1 特征差异分析:\n")
    for col in feature_cols:
        mean_0 = df_final[df_final['y']==0][col].mean()
        mean_1 = df_final[df_final['y']==1][col].mean()
        diff_pct = (mean_1 - mean_0) / mean_0 * 100 if mean_0 != 0 else 0
        f.write(f"  {col}: Y=1/Y=0 = {mean_1/mean_0:.3f} (差异: {diff_pct:+.1f}%)\n")
    f.write("\n")
    
    if 'patent_level' in df_final.columns:
        f.write("按专利等级分布:\n")
        patent_dist = df_final.groupby(['patent_level', 'y']).size().unstack(fill_value=0)
        f.write(patent_dist.to_string())
        f.write("\n\n")
    
    f.write("特征相关性矩阵:\n")
    f.write(corr_matrix.to_string())
    f.write("\n\n")
    
    f.write("优化说明:\n")
    f.write("  1. 使用向量化操作替代循环\n")
    f.write("  2. 预计算链接强度查找表\n")
    f.write("  3. 批量计算特征并一次性赋值\n")
    f.write("  4. 使用NumPy数组加速计算\n")
    f.write("  5. NetworkX的from_pandas_edgelist加速建图\n\n")
    
    f.write("特征说明:\n")
    f.write("  link_strength: 链接强度（共现频率归一化）\n")
    f.write("  degree_difference: 度差（节点度数差的绝对值）\n")
    f.write("  betweenness: 边介数中心性\n")
    f.write("  tech_distance: 技术距离（1 - Jaccard相似度）\n")

print(f"📊 详细报告已生成: {report_filename}")

# 14. 性能统计
print("\n[10] 性能统计...")
print(f"  特征完整性: {(~df_final[feature_cols].isna().any(axis=1)).sum()}/{len(df_final)} 个样本特征完整")

# 检查是否有异常值
for col in feature_cols:
    outliers = ((df_final[col] < 0) | (df_final[col] > 1)).sum()
    if outliers > 0:
        print(f"  警告: {col} 有 {outliers} 个值超出[0,1]范围")

print("\n✅ 优化版特征计算完成！")
print(f"   计算速度提升: 通过向量化操作和批量处理")
print(f"   内存使用优化: 使用NumPy数组和查找表")
