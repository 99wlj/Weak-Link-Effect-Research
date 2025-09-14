"""
X3特征计算脚本 - 关联特征（创新点语义网络 + 发明人合作网络）
使用向量化操作优化性能
"""

import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
import os
import warnings
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import community as community_louvain
import ast
import time
from scipy import sparse
from collections import defaultdict
warnings.filterwarnings('ignore')

# 配置
DATA_DIR = 'F:/WLJ/Weak-Link-Effect-Research/data'

# 输入文件
X1X2_FEATURES_FILE = os.path.join(DATA_DIR, "RES03_features_X1X2_complete_20250914_003935.csv")  # 需要替换为实际文件名
KEYWORDS_FILE = os.path.join(DATA_DIR, "Keywords_Processed_20250903_211104.csv")
INNOVATIONS_FILE = os.path.join(DATA_DIR, "Innovations_Extracted_20250901_172340.csv")
INVENTORS_FILE = os.path.join(DATA_DIR, "TYT_DATA_PERSON_INFO_INVENTOR_ONLY.csv")
LINK_STRENGTH_FILE = os.path.join(DATA_DIR, "Keyword_LinkStrength_ByWindow_20250903_212630.csv")

print("=" * 80)
print("X3特征计算 - 关联特征（创新点语义网络 + 发明人合作网络）")
print("=" * 80)

# 1. 读取数据
print("\n[1] 加载数据...")
df_x1x2 = pd.read_csv(X1X2_FEATURES_FILE)
df_keywords = pd.read_csv(KEYWORDS_FILE)
df_innovations = pd.read_csv(INNOVATIONS_FILE)
df_inventors = pd.read_csv(INVENTORS_FILE)
df_links = pd.read_csv(LINK_STRENGTH_FILE)

print(f"  X1X2特征样本数: {len(df_x1x2):,}")
print(f"  关键词记录数: {len(df_keywords):,}")
print(f"  创新点记录数: {len(df_innovations):,}")
print(f"  发明人记录数: {len(df_inventors):,}")

# 2. 加载语义模型
print("\n[2] 加载语义嵌入模型...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("  模型加载完成: all-MiniLM-L6-v2")

# 3. 数据预处理
print("\n[3] 数据预处理...")

# 解析字符串形式的列表
def safe_eval_list(x):
    """安全地将字符串形式的列表转换为Python列表"""
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except:
        return []

# 处理关键词
df_keywords['keywords'] = df_keywords['keywords'].apply(safe_eval_list)
# 创建关键词到专利的映射（向量化）
keyword_to_patents = defaultdict(set)
for idx, row in df_keywords.iterrows():
    for kw in row['keywords']:
        keyword_to_patents[kw].add(row['appln_id'])

# 处理创新点
df_innovations['innovations'] = df_innovations['innovations'].apply(safe_eval_list)
# 创建专利到创新点的映射
patent_to_innovations = dict(zip(df_innovations['appln_id'], df_innovations['innovations']))

# 创建专利到发明人的映射（向量化）
patent_to_inventors = df_inventors.groupby('appln_id')['person_id'].apply(set).to_dict()

print(f"  关键词数量: {len(keyword_to_patents):,}")
print(f"  包含创新点的专利数: {len(patent_to_innovations):,}")
print(f"  包含发明人的专利数: {len(patent_to_inventors):,}")

# 4. 定义网络特征计算函数
def calculate_network_features(G):
    """计算网络的5个核心特征"""
    features = {}
    
    if G.number_of_nodes() == 0:
        return {
            'density': 0, 
            'avg_clustering': 0, 
            'diameter': 0, 
            'degree_centrality_std': 0, 
            'modularity': 0
        }
    
    # 1. 网络密度
    features['density'] = nx.density(G) if G.number_of_nodes() > 1 else 0
    
    # 2. 平均聚类系数
    features['avg_clustering'] = nx.average_clustering(G) if G.number_of_nodes() > 0 else 0
    
    # 3. 网络直径（使用最大连通分量）
    if G.number_of_nodes() > 1:
        components = list(nx.connected_components(G))
        if components:
            largest_cc = G.subgraph(max(components, key=len))
            if largest_cc.number_of_nodes() > 1:
                features['diameter'] = nx.diameter(largest_cc)
            else:
                features['diameter'] = 0
        else:
            features['diameter'] = 0
    else:
        features['diameter'] = 0
    
    # 4. 度中心性标准差
    if G.number_of_nodes() > 1:
        degree_centrality = nx.degree_centrality(G)
        features['degree_centrality_std'] = np.std(list(degree_centrality.values()))
    else:
        features['degree_centrality_std'] = 0
    
    # 5. 模块度
    if G.number_of_edges() > 0:
        try:
            partition = community_louvain.best_partition(G)
            features['modularity'] = community_louvain.modularity(partition, G)
        except:
            features['modularity'] = 0
    else:
        features['modularity'] = 0
    
    return features

# 5. 批量计算创新点embedding（向量化）
print("\n[4] 计算创新点嵌入向量...")

# 收集所有创新点文本
all_innovations_text = []
innovation_to_idx = {}
idx_counter = 0

for appln_id, innovations in patent_to_innovations.items():
    for innovation in innovations:
        if innovation not in innovation_to_idx:
            innovation_to_idx[innovation] = idx_counter
            all_innovations_text.append(innovation)
            idx_counter += 1

print(f"  独特创新点数量: {len(all_innovations_text):,}")

# 批量计算embedding（向量化操作）
if all_innovations_text:
    print("  批量计算embedding...")
    batch_size = 256
    all_embeddings = []
    
    for i in range(0, len(all_innovations_text), batch_size):
        batch = all_innovations_text[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.append(batch_embeddings)
    
    innovation_embeddings = np.vstack(all_embeddings)
    print(f"  Embedding矩阵形状: {innovation_embeddings.shape}")
else:
    innovation_embeddings = np.array([])

# 6. 初始化X3特征数组
n_samples = len(df_x1x2)
print(f"\n[5] 初始化X3特征数组 (样本数: {n_samples:,})...")

# 创新点网络特征
innovation_density = np.zeros(n_samples)
innovation_avg_clustering = np.zeros(n_samples)
innovation_diameter = np.zeros(n_samples)
innovation_degree_centrality_std = np.zeros(n_samples)
innovation_modularity = np.zeros(n_samples)

# 发明人网络特征
inventor_density = np.zeros(n_samples)
inventor_avg_clustering = np.zeros(n_samples)
inventor_diameter = np.zeros(n_samples)
inventor_degree_centrality_std = np.zeros(n_samples)
inventor_modularity = np.zeros(n_samples)

# 7. 批量处理每个样本（边）
print("\n[6] 批量计算X3特征...")

# 创建时间窗口标识
df_x1x2['window_id'] = df_x1x2['window_t_start'].astype(str) + '-' + df_x1x2['window_t_end'].astype(str)

# 按时间窗口分组处理
unique_windows = df_x1x2['window_id'].unique()
print(f"  处理 {len(unique_windows)} 个时间窗口...")

for window_idx, window_id in enumerate(unique_windows, 1):
    print(f"\n  窗口 {window_idx}/{len(unique_windows)}: {window_id}")
    window_start_time = time.time()
    
    # 获取该窗口的样本
    window_mask = df_x1x2['window_id'] == window_id
    window_samples = df_x1x2[window_mask]
    window_indices = window_samples.index.values
    
    print(f"    样本数: {len(window_samples):,}")
    
    # 批量处理该窗口的所有样本
    for idx in window_indices:
        node_u = df_x1x2.loc[idx, 'node_u']
        node_v = df_x1x2.loc[idx, 'node_v']
        
        # 获取边对应的专利集合（两个关键词的专利交集）
        patents_u = keyword_to_patents.get(node_u, set())
        patents_v = keyword_to_patents.get(node_v, set())
        edge_patents = patents_u & patents_v
        
        if len(edge_patents) == 0:
            continue
        
        # === 创新点语义网络特征 ===
        edge_innovations = []
        edge_innovation_indices = []
        
        for patent_id in edge_patents:
            if patent_id in patent_to_innovations:
                for innovation in patent_to_innovations[patent_id]:
                    if innovation in innovation_to_idx:
                        edge_innovations.append(innovation)
                        edge_innovation_indices.append(innovation_to_idx[innovation])
        
        if len(edge_innovations) > 1:
            # 提取相关的embedding（向量化）
            edge_embeddings = innovation_embeddings[edge_innovation_indices]
            
            # 计算相似度矩阵（向量化）
            similarity_matrix = cosine_similarity(edge_embeddings)
            
            # 构建创新点相似网络（设置阈值）
            threshold = 0.5
            G_innovation = nx.Graph()
            G_innovation.add_nodes_from(range(len(edge_innovations)))
            
            # 向量化添加边
            edges_to_add = []
            for i in range(len(edge_innovations)):
                for j in range(i+1, len(edge_innovations)):
                    if similarity_matrix[i, j] > threshold:
                        edges_to_add.append((i, j, {'weight': similarity_matrix[i, j]}))
            
            G_innovation.add_edges_from(edges_to_add)
            
            # 计算网络特征
            innovation_features = calculate_network_features(G_innovation)
            
            # 赋值特征
            innovation_density[idx] = innovation_features['density']
            innovation_avg_clustering[idx] = innovation_features['avg_clustering']
            innovation_diameter[idx] = innovation_features['diameter']
            innovation_degree_centrality_std[idx] = innovation_features['degree_centrality_std']
            innovation_modularity[idx] = innovation_features['modularity']
        
        # === 发明人合作网络特征 ===
        edge_inventors = set()
        inventor_collaborations = defaultdict(set)
        
        for patent_id in edge_patents:
            if patent_id in patent_to_inventors:
                patent_inventors = patent_to_inventors[patent_id]
                edge_inventors.update(patent_inventors)
                
                # 记录合作关系（同一专利的发明人之间存在合作）
                for inv1 in patent_inventors:
                    for inv2 in patent_inventors:
                        if inv1 != inv2:
                            inventor_collaborations[inv1].add(inv2)
        
        if len(edge_inventors) > 1:
            # 构建发明人网络
            G_inventor = nx.Graph()
            G_inventor.add_nodes_from(edge_inventors)
            
            # 向量化添加边
            edges_to_add = []
            for inv1, collaborators in inventor_collaborations.items():
                for inv2 in collaborators:
                    if inv1 < inv2:  # 避免重复添加边
                        edges_to_add.append((inv1, inv2))
            
            G_inventor.add_edges_from(edges_to_add)
            
            # 计算网络特征
            inventor_features = calculate_network_features(G_inventor)
            
            # 赋值特征
            inventor_density[idx] = inventor_features['density']
            inventor_avg_clustering[idx] = inventor_features['avg_clustering']
            inventor_diameter[idx] = inventor_features['diameter']
            inventor_degree_centrality_std[idx] = inventor_features['degree_centrality_std']
            inventor_modularity[idx] = inventor_features['modularity']
    
    elapsed = time.time() - window_start_time
    print(f"    耗时: {elapsed:.2f}秒")

# 8. 添加X3特征到DataFrame
print("\n[7] 整合X3特征...")

# 创新点网络特征
df_x1x2['innovation_density'] = innovation_density
df_x1x2['innovation_avg_clustering'] = innovation_avg_clustering
df_x1x2['innovation_diameter'] = innovation_diameter
df_x1x2['innovation_degree_centrality_std'] = innovation_degree_centrality_std
df_x1x2['innovation_modularity'] = innovation_modularity

# 发明人网络特征
df_x1x2['inventor_density'] = inventor_density
df_x1x2['inventor_avg_clustering'] = inventor_avg_clustering
df_x1x2['inventor_diameter'] = inventor_diameter
df_x1x2['inventor_degree_centrality_std'] = inventor_degree_centrality_std
df_x1x2['inventor_modularity'] = inventor_modularity

# 9. X3特征标准化
print("\n[8] X3特征标准化...")

x3_feature_cols = [
    'innovation_density', 'innovation_avg_clustering', 'innovation_diameter',
    'innovation_degree_centrality_std', 'innovation_modularity',
    'inventor_density', 'inventor_avg_clustering', 'inventor_diameter',
    'inventor_degree_centrality_std', 'inventor_modularity'
]

# 保存原始值
for col in x3_feature_cols:
    df_x1x2[f'{col}_raw'] = df_x1x2[col].values

# 标准化到[0,1]
scaler = MinMaxScaler()
for col in x3_feature_cols:
    col_values = df_x1x2[col].values.reshape(-1, 1)
    
    # 检查是否有有效的方差
    if np.std(col_values) > 1e-10:
        df_x1x2[col] = scaler.fit_transform(col_values).flatten()
    else:
        print(f"  警告: {col} 方差过小，保持原值")
        if np.mean(col_values) > 0:
            df_x1x2[col] = 1.0
        else:
            df_x1x2[col] = 0.0

print("  X3特征已标准化到 [0, 1] 区间")

# 10. 整理最终数据（X1 + X2 + X3）
print("\n[9] 整理最终数据（X1 + X2 + X3）...")

# 确定所有特征列
x1_feature_cols = ['link_strength', 'degree_difference', 'betweenness', 'tech_distance']
x2_feature_cols = ['common_neighbors', 'adamic_adar', 'resource_allocation', 'preferential_attachment']
all_feature_cols = x1_feature_cols + x2_feature_cols + x3_feature_cols

# 移除临时列
if 'window_id' in df_x1x2.columns:
    df_x1x2.drop('window_id', axis=1, inplace=True)

# 11. 保存结果
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"RES04_features_X1X2X3_complete_{timestamp}.csv"
output_path = os.path.join(DATA_DIR, output_filename)

df_x1x2.to_csv(output_path, index=False, encoding='utf-8-sig')

print("\n" + "=" * 80)
print(f"✅ X1+X2+X3特征计算完成！")
print(f"   输出文件: {output_filename}")
print(f"   路径: {output_path}")
print(f"   样本数: {len(df_x1x2):,}")
print(f"   特征总数: {len(all_feature_cols)} 个")
print(f"     - X1特征: {len(x1_feature_cols)} 个")
print(f"     - X2特征: {len(x2_feature_cols)} 个")
print(f"     - X3特征: {len(x3_feature_cols)} 个")
print("=" * 80)

# 12. 生成统计报告
print("\n[10] 生成统计报告...")

# X3特征统计
print("\nX3特征统计（标准化后）:")
print(df_x1x2[x3_feature_cols].describe())

# 按Y值分组统计
print("\n按Y值分组的X3特征均值（标准化后）:")
grouped_x3 = df_x1x2.groupby('y')[x3_feature_cols].mean()
print(grouped_x3)

# 特征相关性
print("\nX3特征内部相关性:")
corr_matrix_x3 = df_x1x2[x3_feature_cols].corr()
print(corr_matrix_x3)

# 13. 生成详细报告文件
report_filename = f"RES04_features_X1X2X3_report_{timestamp}.txt"
report_path = os.path.join(DATA_DIR, report_filename)

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("X1+X2+X3完整特征计算报告\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("数据概况:\n")
    f.write(f"  样本总数: {len(df_x1x2):,}\n")
    f.write(f"  正样本数: {(df_x1x2['y']==1).sum():,}\n")
    f.write(f"  负样本数: {(df_x1x2['y']==0).sum():,}\n")
    f.write(f"  特征总数: {len(all_feature_cols)}\n")
    f.write(f"    - X1特征: {len(x1_feature_cols)}个\n")
    f.write(f"    - X2特征: {len(x2_feature_cols)}个\n")
    f.write(f"    - X3特征: {len(x3_feature_cols)}个\n\n")
    
    f.write("X3特征说明:\n")
    f.write("创新点语义网络特征:\n")
    f.write("  - innovation_density: 创新点网络密度\n")
    f.write("  - innovation_avg_clustering: 创新点网络平均聚类系数\n")
    f.write("  - innovation_diameter: 创新点网络直径\n")
    f.write("  - innovation_degree_centrality_std: 创新点网络度中心性标准差\n")
    f.write("  - innovation_modularity: 创新点网络模块度\n\n")
    
    f.write("发明人合作网络特征:\n")
    f.write("  - inventor_density: 发明人网络密度\n")
    f.write("  - inventor_avg_clustering: 发明人网络平均聚类系数\n")
    f.write("  - inventor_diameter: 发明人网络直径\n")
    f.write("  - inventor_degree_centrality_std: 发明人网络度中心性标准差\n")
    f.write("  - inventor_modularity: 发明人网络模块度\n\n")
    
    f.write("X3特征统计（标准化后）:\n")
    f.write(df_x1x2[x3_feature_cols].describe().to_string())
    f.write("\n\n")
    
    f.write("按Y值分组的X3特征均值对比:\n")
    f.write(grouped_x3.to_string())
    f.write("\n\n")
    
    # Y=0 vs Y=1 的差异分析
    f.write("Y=0 vs Y=1 X3特征差异分析:\n")
    for col in x3_feature_cols:
        mean_0 = df_x1x2[df_x1x2['y']==0][col].mean()
        mean_1 = df_x1x2[df_x1x2['y']==1][col].mean()
        if mean_0 != 0:
            ratio = mean_1 / mean_0
            diff_pct = (mean_1 - mean_0) / mean_0 * 100
            f.write(f"  {col}: Y=1/Y=0 = {ratio:.3f} (差异: {diff_pct:+.1f}%)\n")
        else:
            f.write(f"  {col}: Y=0均值为0\n")
    
    f.write("\n")
    
    # 特征重要性排序（基于Y=1和Y=0的差异）
    f.write("X3特征重要性排序（基于Y值差异）:\n")
    feature_importance = []
    for col in x3_feature_cols:
        mean_0 = df_x1x2[df_x1x2['y']==0][col].mean()
        mean_1 = df_x1x2[df_x1x2['y']==1][col].mean()
        diff = abs(mean_1 - mean_0)
        feature_importance.append((col, diff))
    
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    for i, (col, diff) in enumerate(feature_importance, 1):
        f.write(f"  {i}. {col}: 差异值 = {diff:.4f}\n")
    
    f.write("\n")
    
    if 'patent_level' in df_x1x2.columns:
        f.write("按专利等级的X3特征均值:\n")
        patent_x3_stats = df_x1x2.groupby('patent_level')[x3_feature_cols].mean()
        f.write(patent_x3_stats.to_string())
        f.write("\n\n")
    
    f.write("X3特征内部相关性矩阵:\n")
    f.write(corr_matrix_x3.to_string())
    f.write("\n\n")
    
    # 高相关特征对
    f.write("X3高相关特征对（|r| > 0.7）:\n")
    high_corr_pairs = []
    for i in range(len(x3_feature_cols)):
        for j in range(i+1, len(x3_feature_cols)):
            corr_val = corr_matrix_x3.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append((x3_feature_cols[i], x3_feature_cols[j], corr_val))
    
    if high_corr_pairs:
        for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            f.write(f"  {feat1} <-> {feat2}: {corr:.3f}\n")
    else:
        f.write("  无高相关特征对\n")
    
    f.write("\n计算方法说明:\n")
    f.write("1. 对每条边（关键词对），找到包含这两个关键词的专利集合\n")
    f.write("2. 基于专利集合的创新点，构建语义相似网络（相似度>0.5）\n")
    f.write("3. 基于专利集合的发明人，构建合作网络\n")
    f.write("4. 计算两个网络的拓扑特征指标\n")
    f.write("5. 标准化到[0,1]区间\n\n")
    
    f.write("优化技术:\n")
    f.write("  1. 批量计算embedding（向量化）\n")
    f.write("  2. 使用cosine_similarity批量计算相似度\n")
    f.write("  3. 向量化的边添加操作\n")
    f.write("  4. 预计算和缓存映射关系\n")
    f.write("  5. 按时间窗口批处理\n")

print(f"📊 详细报告已生成: {report_filename}")

# 14. 数据完整性检查
print("\n[11] 数据完整性检查...")

# 检查缺失值
missing_features = df_x1x2[all_feature_cols].isna().sum()
if missing_features.sum() > 0:
    print("  警告：存在缺失值")
    print(missing_features[missing_features > 0])
else:
    print("  ✓ 所有特征值完整")

# 检查异常值
for col in x3_feature_cols:
    outliers = ((df_x1x2[col] < 0) | (df_x1x2[col] > 1)).sum()
    if outliers > 0:
        print(f"  警告: {col} 有 {outliers} 个值超出[0,1]范围")

# 特征分布检查
print("\nX3特征值分布检查:")
for col in x3_feature_cols:
    non_zero = (df_x1x2[col] > 0).sum()
    zero_pct = (1 - non_zero/len(df_x1x2)) * 100
    print(f"  {col}: {zero_pct:.1f}%为0值, {non_zero:,}个非零值")

print("\n✅ X1+X2+X3完整特征集计算完成！")
print(f"   X3特征亮点: ")
print(f"   - 创新点语义网络: 基于embedding相似度构建")
print(f"   - 发明人合作网络: 基于专利合作关系构建")
print(f"   - 10个网络拓扑特征: 密度、聚类、直径、中心性、模块度")
print(f"   输出文件: {output_filename}")
