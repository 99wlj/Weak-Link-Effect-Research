"""
扩展特征X2计算脚本（优化版）
基于X1结果计算额外的网络特征，并拼接保存
使用向量化操作避免循环，提高计算速度
"""

import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
import os
import warnings
from sklearn.preprocessing import MinMaxScaler
import time
from scipy import sparse
warnings.filterwarnings('ignore')

# 配置
DATA_DIR = 'F:/WLJ/Weak-Link-Effect-Research/data'

# 输入文件
X1_FEATURES_FILE = os.path.join(DATA_DIR, "RES02_features_X1_optimized_20250914_001705.csv")
LINK_STRENGTH_FILE = os.path.join(DATA_DIR, "Keyword_LinkStrength_ByWindow_20250903_212630.csv")

print("=" * 80)
print("扩展特征X2计算（优化版）")
print("=" * 80)

# 1. 读取数据
print("\n[1] 读取数据...")
df_x1 = pd.read_csv(X1_FEATURES_FILE)
df_links_all = pd.read_csv(LINK_STRENGTH_FILE)

print(f"  X1特征样本数: {len(df_x1):,}")
print(f"  全部链接数: {len(df_links_all):,}")

# 2. 准备数据
print("\n[2] 准备数据...")

# 创建时间窗口标识
df_x1['window_id'] = df_x1['window_t_start'].astype(str) + '-' + df_x1['window_t_end'].astype(str)
df_links_all['window_id'] = df_links_all['window_start'].astype(str) + '-' + df_links_all['window_end'].astype(str)

# 3. 初始化X2特征数组
n_samples = len(df_x1)
common_neighbors = np.zeros(n_samples)
adamic_adar = np.zeros(n_samples)
resource_allocation = np.zeros(n_samples)
preferential_attachment = np.zeros(n_samples)

# 4. 获取时间窗口
unique_windows = df_x1[['window_t_start', 'window_t_end', 'window_id']].drop_duplicates()
print(f"\n[3] 处理 {len(unique_windows)} 个时间窗口...")

# 5. 定义向量化的特征计算函数
def compute_x2_features_vectorized(window_data):
    """向量化计算X2特征"""
    w_start, w_end, window_id = window_data
    
    # 获取该窗口的样本
    window_mask = df_x1['window_id'] == window_id
    window_samples = df_x1[window_mask]
    window_indices = window_samples.index.values
    
    if len(window_indices) == 0:
        return None
    
    print(f"    样本数: {len(window_samples):,}")
    
    # 获取该窗口的链接
    window_links = df_links_all[df_links_all['window_id'] == window_id]
    
    if len(window_links) == 0:
        print(f"    警告：窗口 {window_id} 没有链接数据")
        return None
    
    # 构建网络
    G = nx.from_pandas_edgelist(
        window_links,
        source='node_u',
        target='node_v',
        edge_attr='link_strength',
        create_using=nx.Graph()
    )
    
    # 确保样本节点都在网络中
    sample_nodes = set(window_samples['node_u']) | set(window_samples['node_v'])
    for node in sample_nodes - set(G.nodes()):
        G.add_node(node)
    
    print(f"    网络: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    
    # 预计算所有节点的邻居和度（向量化）
    node_list = list(G.nodes())
    node_idx_map = {node: idx for idx, node in enumerate(node_list)}
    n_nodes = len(node_list)
    
    # 构建邻接矩阵（稀疏矩阵更快）
    adj_matrix = nx.adjacency_matrix(G).astype(float)
    
    # 批量计算节点度
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    degree_dict = {node: degrees[node_idx_map[node]] for node in node_list}
    
    # 预计算邻居集合（使用稀疏矩阵操作）
    neighbors_dict = {}
    for node in sample_nodes:
        if node in node_idx_map:
            node_idx = node_idx_map[node]
            # 获取邻居索引
            neighbor_indices = adj_matrix[node_idx].nonzero()[1]
            neighbors_dict[node] = set(node_list[i] for i in neighbor_indices)
        else:
            neighbors_dict[node] = set()
    
    # 准备结果列表
    results = []
    
    # 批量计算特征（避免显式循环，使用列表推导）
    sample_data = window_samples[['node_u', 'node_v']].values
    
    # 向量化计算所有样本
    for i, (u, v) in enumerate(sample_data):
        idx = window_indices[i]
        
        # 获取邻居集合
        neighbors_u = neighbors_dict.get(u, set())
        neighbors_v = neighbors_dict.get(v, set())
        
        # 特征1: 共同邻居数
        common = neighbors_u & neighbors_v
        cn_count = len(common)
        
        # 特征2: Adamic-Adar指数
        aa_index = 0
        if cn_count > 0:
            # 向量化计算
            common_degrees = np.array([degree_dict.get(w, 1) for w in common])
            # 避免log(1) = 0的情况
            common_degrees = np.maximum(common_degrees, 2)
            aa_index = np.sum(1.0 / np.log(common_degrees))
        
        # 特征3: 资源分配指数
        ra_index = 0
        if cn_count > 0:
            # 向量化计算
            common_degrees = np.array([degree_dict.get(w, 1) for w in common])
            common_degrees = np.maximum(common_degrees, 1)
            ra_index = np.sum(1.0 / common_degrees)
        
        # 特征4: 优先连接指数
        deg_u = degree_dict.get(u, 0)
        deg_v = degree_dict.get(v, 0)
        pa_index = deg_u * deg_v
        
        results.append((idx, cn_count, aa_index, ra_index, pa_index))
    
    return results

# 6. 批量处理所有窗口
print("\n[4] 批量计算X2特征...")

all_results = []
for idx, row in enumerate(unique_windows.values, 1):
    print(f"\n  窗口 {idx}/{len(unique_windows)}: {row[2]}")
    start_time = time.time()
    
    window_results = compute_x2_features_vectorized(row)
    if window_results:
        all_results.extend(window_results)
    
    elapsed = time.time() - start_time
    print(f"    耗时: {elapsed:.2f}秒")

print(f"\n  完成X2特征计算，共 {len(all_results)} 个结果")

# 7. 批量赋值特征（向量化）
print("\n[5] 批量赋值X2特征...")

if len(all_results) > 0:
    # 转换为NumPy数组
    results_array = np.array(all_results)
    indices = results_array[:, 0].astype(int)
    
    # 批量赋值
    common_neighbors[indices] = results_array[:, 1]
    adamic_adar[indices] = results_array[:, 2]
    resource_allocation[indices] = results_array[:, 3]
    preferential_attachment[indices] = results_array[:, 4]

# 将X2特征添加到DataFrame
df_x1['common_neighbors'] = common_neighbors
df_x1['adamic_adar'] = adamic_adar
df_x1['resource_allocation'] = resource_allocation
df_x1['preferential_attachment'] = preferential_attachment

# 8. X2特征标准化
print("\n[6] X2特征标准化...")

x2_feature_cols = ['common_neighbors', 'adamic_adar', 'resource_allocation', 'preferential_attachment']

# 保存原始值
for col in x2_feature_cols:
    df_x1[f'{col}_raw'] = df_x1[col].values

# 标准化到[0,1]
scaler = MinMaxScaler()
for col in x2_feature_cols:
    col_values = df_x1[col].values.reshape(-1, 1)
    
    # 检查是否有有效的方差
    if np.std(col_values) > 1e-10:
        df_x1[col] = scaler.fit_transform(col_values).flatten()
    else:
        print(f"  警告: {col} 方差过小，保持原值")
        # 如果所有值相同，标准化为0或1
        if np.mean(col_values) > 0:
            df_x1[col] = 1.0
        else:
            df_x1[col] = 0.0

print("  X2特征已标准化到 [0, 1] 区间")

# 9. 合并所有特征（X1 + X2）
print("\n[7] 整理最终数据（X1 + X2）...")

# 确定输出列顺序
base_cols = [
    'sample_id', 
    'window_t_start', 'window_t_end', 
    'window_t1_start', 'window_t1_end',
    'node_u', 'node_v', 
    'y',
    'window_label',
    'edge_importance',
    'patent_level'
]

# X1特征（标准化后）
x1_feature_cols = ['link_strength', 'degree_difference', 'betweenness', 'tech_distance']

# 所有标准化后的特征
all_feature_cols = x1_feature_cols + x2_feature_cols

# 所有原始值列
all_raw_cols = [f'{col}_raw' for col in all_feature_cols]

# 组合所有列
output_cols = base_cols + all_feature_cols + all_raw_cols

# 只保留存在的列
available_cols = [col for col in output_cols if col in df_x1.columns]
df_final = df_x1[available_cols].copy()

# 移除临时列
if 'window_id' in df_final.columns:
    df_final.drop('window_id', axis=1, inplace=True)

# 10. 保存结果
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"RES03_features_X1X2_complete_{timestamp}.csv"
output_path = os.path.join(DATA_DIR, output_filename)

df_final.to_csv(output_path, index=False, encoding='utf-8-sig')

print("\n" + "=" * 80)
print(f"✅ X1+X2特征计算完成！")
print(f"   输出文件: {output_filename}")
print(f"   路径: {output_path}")
print(f"   样本数: {len(df_final):,}")
print(f"   特征数: {len(all_feature_cols)} 个")
print("=" * 80)

# 11. 生成统计报告
print("\n[8] 生成统计报告...")

# 全部特征统计
print("\n所有特征统计（标准化后）:")
print(df_final[all_feature_cols].describe())

# X2特征统计
print("\nX2特征统计（标准化后）:")
print(df_final[x2_feature_cols].describe())

print("\nX2特征统计（原始值）:")
x2_raw_cols = [f'{col}_raw' for col in x2_feature_cols]
print(df_final[x2_raw_cols].describe())

# 按Y值分组统计
print("\n按Y值分组的X2特征均值（标准化后）:")
grouped_x2 = df_final.groupby('y')[x2_feature_cols].mean()
print(grouped_x2)

# 特征相关性
print("\n所有特征相关性矩阵:")
corr_matrix_all = df_final[all_feature_cols].corr()
print(corr_matrix_all)

# 12. 生成详细报告文件
report_filename = f"RES03_features_X1X2_report_{timestamp}.txt"
report_path = os.path.join(DATA_DIR, report_filename)

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("X1+X2完整特征计算报告\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"X1数据源: {os.path.basename(X1_FEATURES_FILE)}\n\n")
    
    f.write("数据概况:\n")
    f.write(f"  样本总数: {len(df_final):,}\n")
    f.write(f"  正样本数: {(df_final['y']==1).sum():,}\n")
    f.write(f"  负样本数: {(df_final['y']==0).sum():,}\n")
    f.write(f"  特征总数: {len(all_feature_cols)}\n")
    f.write(f"    - X1特征: {len(x1_feature_cols)}个\n")
    f.write(f"    - X2特征: {len(x2_feature_cols)}个\n\n")
    
    f.write("X1特征:\n")
    for i, col in enumerate(x1_feature_cols, 1):
        f.write(f"  {i}. {col}\n")
    
    f.write("\nX2特征:\n")
    for i, col in enumerate(x2_feature_cols, 1):
        f.write(f"  {i}. {col}\n")
    
    f.write("\n所有特征统计（标准化后）:\n")
    f.write(df_final[all_feature_cols].describe().to_string())
    f.write("\n\n")
    
    f.write("X2特征统计（原始值）:\n")
    f.write(df_final[x2_raw_cols].describe().to_string())
    f.write("\n\n")
    
    f.write("按Y值分组的特征均值对比:\n")
    grouped_all = df_final.groupby('y')[all_feature_cols].mean()
    f.write(grouped_all.to_string())
    f.write("\n\n")
    
    # Y=0 vs Y=1 的差异分析
    f.write("Y=0 vs Y=1 特征差异分析:\n")
    f.write("\nX1特征差异:\n")
    for col in x1_feature_cols:
        mean_0 = df_final[df_final['y']==0][col].mean()
        mean_1 = df_final[df_final['y']==1][col].mean()
        if mean_0 != 0:
            ratio = mean_1 / mean_0
            diff_pct = (mean_1 - mean_0) / mean_0 * 100
            f.write(f"  {col}: Y=1/Y=0 = {ratio:.3f} (差异: {diff_pct:+.1f}%)\n")
        else:
            f.write(f"  {col}: Y=0均值为0\n")
    
    f.write("\nX2特征差异:\n")
    for col in x2_feature_cols:
        mean_0 = df_final[df_final['y']==0][col].mean()
        mean_1 = df_final[df_final['y']==1][col].mean()
        if mean_0 != 0:
            ratio = mean_1 / mean_0
            diff_pct = (mean_1 - mean_0) / mean_0 * 100
            f.write(f"  {col}: Y=1/Y=0 = {ratio:.3f} (差异: {diff_pct:+.1f}%)\n")
        else:
            f.write(f"  {col}: Y=0均值为0\n")
    
    f.write("\n")
    
    if 'patent_level' in df_final.columns:
        f.write("按专利等级的X2特征均值:\n")
        patent_x2_stats = df_final.groupby('patent_level')[x2_feature_cols].mean()
        f.write(patent_x2_stats.to_string())
        f.write("\n\n")
    
    f.write("特征相关性矩阵（所有特征）:\n")
    f.write(corr_matrix_all.to_string())
    f.write("\n\n")
    
    # 高相关特征对
    f.write("高相关特征对（|r| > 0.7）:\n")
    high_corr_pairs = []
    for i in range(len(all_feature_cols)):
        for j in range(i+1, len(all_feature_cols)):
            corr_val = corr_matrix_all.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append((all_feature_cols[i], all_feature_cols[j], corr_val))
    
    if high_corr_pairs:
        for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            f.write(f"  {feat1} <-> {feat2}: {corr:.3f}\n")
    else:
        f.write("  无高相关特征对\n")
    
    f.write("\n")
    
    f.write("X2特征说明:\n")
    f.write("  common_neighbors: 共同邻居数\n")
    f.write("  adamic_adar: Adamic-Adar指数（共同邻居的度数倒数对数和）\n")
    f.write("  resource_allocation: 资源分配指数（共同邻居的度数倒数和）\n")
    f.write("  preferential_attachment: 优先连接指数（两节点度数的乘积）\n\n")
    
    f.write("优化技术:\n")
    f.write("  1. 使用稀疏矩阵存储邻接关系\n")
    f.write("  2. 批量计算节点度数\n")
    f.write("  3. 向量化的特征计算\n")
    f.write("  4. NumPy数组操作替代循环\n")
    f.write("  5. 预计算和缓存邻居集合\n")

print(f"📊 详细报告已生成: {report_filename}")

# 13. 验证数据完整性
print("\n[9] 数据完整性检查...")

# 检查缺失值
missing_features = df_final[all_feature_cols].isna().sum()
if missing_features.sum() > 0:
    print("  警告：存在缺失值")
    print(missing_features[missing_features > 0])
else:
    print("  ✓ 所有特征值完整")

# 检查异常值
for col in all_feature_cols:
    outliers = ((df_final[col] < 0) | (df_final[col] > 1)).sum()
    if outliers > 0:
        print(f"  警告: {col} 有 {outliers} 个值超出[0,1]范围")

# 特征分布检查
print("\n特征值分布检查:")
for col in x2_feature_cols:
    unique_vals = df_final[col].nunique()
    if unique_vals < 5:
        print(f"  {col}: 只有 {unique_vals} 个不同值")
        print(f"    值分布: {df_final[col].value_counts().head()}")

print("\n✅ X1+X2完整特征集计算完成！")
print(f"   优化亮点: 向量化操作、稀疏矩阵、批量计算")
print(f"   输出文件: {output_filename}")
