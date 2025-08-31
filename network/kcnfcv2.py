import pandas as pd
import networkx as nx
import numpy as np
from datetime import datetime
import logging
import os
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义输入文件路径
data_dir = "data"
input_file = "keyword_co_network_20250830_154905.csv"
input_path = os.path.join(data_dir, input_file)

# 读取边列表数据
logger.info(f"读取边列表数据: {input_path}")
edges_df = pd.read_csv(input_path)

# 构建网络图
logger.info("构建网络图...")
G = nx.Graph()

# 添加边（同时会添加节点）
for _, row in edges_df.iterrows():
    G.add_edge(row['source'], row['target'], weight=row['weight'])

# 计算网络特征函数
def calculate_edge_features(G, edges_df):
    """
    根据文献定义计算每条边的四个网络特征
    """
    features = []
    
    # 预计算全局网络指标
    logger.info("计算边介数中心性...")
    edge_betweenness = nx.edge_betweenness_centrality(G, weight='weight')
    
    logger.info("计算节点度中心性...")
    node_degrees = dict(G.degree())
    node_strengths = dict(G.degree(weight='weight'))  # 加权度（节点总权重）
    
    logger.info("计算节点特征向量中心性...")
    # 使用特征向量中心性作为节点的全局重要性指标
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        # 如果特征向量中心性计算失败，使用度中心性作为替代
        logger.warning("特征向量中心性计算失败，使用度中心性替代")
        eigenvector_centrality = nx.degree_centrality(G)
    
    logger.info("计算节点PageRank...")
    # 使用PageRank作为节点的另一种全局重要性指标
    pagerank = nx.pagerank(G)
    
    logger.info("计算节点聚类系数...")
    # 使用聚类系数作为节点局部结构的指标
    clustering = nx.clustering(G)
    
    logger.info("计算共同邻居信息...")
    # 预计算所有节点的共同邻居
    common_neighbors_dict = {}
    for edge in G.edges():
        u, v = edge
        common_neighbors_dict[(u, v)] = len(list(nx.common_neighbors(G, u, v)))
    
    for idx, row in edges_df.iterrows():
        source = row['source']
        target = row['target']
        weight = row['weight']
        
        # 1. 链路强度 - 按照文献公式计算
        # 公式: Weighted Tie Strength = (∑(wk + wj)) / (si + sj - 2wij)
        # 获取共同邻居
        common_neighbors = list(nx.common_neighbors(G, source, target))
        
        # 计算分子: 所有共同邻居的权重之和
        common_neighbors_weight_sum = 0
        for neighbor in common_neighbors:
            # 这里采用保守做法：使用共同邻居的加权度作为其权重贡献
            common_neighbors_weight_sum += node_strengths.get(neighbor, 0)
        
        # 分子 = 共同邻居权重和 + 当前边的权重
        numerator = common_neighbors_weight_sum + weight
        
        # 分母 = 两个节点的度之和 - 2倍当前边的权重
        denominator = node_degrees.get(source, 0) + node_degrees.get(target, 0) - 2 * weight
        
        # 避免除以零
        if denominator <= 0:
            weighted_tie_strength = 0
        else:
            weighted_tie_strength = numerator / denominator
        
        # 2. 链路中心性 - 边介数中心性
        edge_key = (min(source, target), max(source, target))
        link_centrality = edge_betweenness.get(edge_key, 0)
        
        # 3. 技术距离 - 使用多种节点指标的差异组合
        # 特征向量中心性差异
        source_eigen = eigenvector_centrality.get(source, 0)
        target_eigen = eigenvector_centrality.get(target, 0)
        eigen_distance = abs(source_eigen - target_eigen)
        
        # PageRank差异
        source_pagerank = pagerank.get(source, 0)
        target_pagerank = pagerank.get(target, 0)
        pagerank_distance = abs(source_pagerank - target_pagerank)
        
        # 聚类系数差异
        source_clustering = clustering.get(source, 0)
        target_clustering = clustering.get(target, 0)
        clustering_distance = abs(source_clustering - target_clustering)
        
        # 综合技术距离（多种指标的加权平均）
        technical_distance = (eigen_distance + pagerank_distance + clustering_distance) / 3
        
        # 4. 节点同配性 - 使用节点度的绝对差值
        source_degree = node_degrees.get(source, 0)
        target_degree = node_degrees.get(target, 0)
        node_assortativity = abs(source_degree - target_degree)
        
        # 5. 共同邻居数量（网络结构特征）
        common_neighbors_count = len(common_neighbors)
        
        features.append({
            'source': source,
            'target': target,
            'weight': weight,
            'weighted_tie_strength': weighted_tie_strength,  
            'link_centrality': link_centrality,
            'technical_distance': technical_distance,
            'node_assortativity': node_assortativity,
            'common_neighbors_count': common_neighbors_count,  # 共同邻居数量
            'eigen_distance': eigen_distance,
            'pagerank_distance': pagerank_distance,
            'clustering_distance': clustering_distance
        })
    
    return pd.DataFrame(features)

# 计算边特征
logger.info("开始计算边特征...")
edge_features_df = calculate_edge_features(G, edges_df)
logger.info("边特征计算完成")

# 根据中位数划分强弱链接
logger.info("根据中位数划分强弱链接...")
median_strength = edge_features_df['weighted_tie_strength'].median()
edge_features_df['link_strength_category'] = edge_features_df['weighted_tie_strength'].apply(
    lambda x: '强链接' if x >= median_strength else '弱链接'
)

# 生成关键词链路强度分类表格
logger.info("生成关键词链路强度分类表格...")
keyword_strength_df = edge_features_df[['source', 'target', 'weighted_tie_strength', 'link_strength_category']].copy()
keyword_strength_df = keyword_strength_df.rename(columns={
    'source': '关键词1',
    'target': '关键词2',
    'weighted_tie_strength': '链路强度值',
    'link_strength_category': '链接强度分类'
})

# 按链路强度值降序排列
keyword_strength_df = keyword_strength_df.sort_values('链路强度值', ascending=False)

# 生成时间戳
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 保存带特征的边数据
output_filename = f"keyword_co_network_with_features_{timestamp}.csv"
output_path = os.path.join(data_dir, output_filename)
edge_features_df.to_csv(output_path, index=False)
logger.info(f"带特征的共现网络已保存到: {output_path}")

# 保存关键词链路强度分类表格
strength_filename = f"keyword_link_strength_classification_{timestamp}.csv"
strength_path = os.path.join(data_dir, strength_filename)
keyword_strength_df.to_csv(strength_path, index=False, encoding='utf-8-sig')  # 使用utf-8-sig支持中文
logger.info(f"关键词链路强度分类表格已保存到: {strength_path}")

# 统计强弱链接数量
strong_links = len(keyword_strength_df[keyword_strength_df['链接强度分类'] == '强链接'])
weak_links = len(keyword_strength_df[keyword_strength_df['链接强度分类'] == '弱链接'])

logger.info(f"共处理 {len(edge_features_df)} 条边")
logger.info(f"中位数链路强度值: {median_strength:.6f}")
logger.info(f"强链接数量: {strong_links} ({strong_links/len(edge_features_df)*100:.1f}%)")
logger.info(f"弱链接数量: {weak_links} ({weak_links/len(edge_features_df)*100:.1f}%)")

# 打印特征统计信息
logger.info("\n特征统计信息:")
logger.info(f"链路强度范围: {edge_features_df['weighted_tie_strength'].min():.6f} - {edge_features_df['weighted_tie_strength'].max():.6f}")
logger.info(f"共同邻居数量范围: {edge_features_df['common_neighbors_count'].min()} - {edge_features_df['common_neighbors_count'].max()}")
logger.info(f"链路中心性范围: {edge_features_df['link_centrality'].min():.6f} - {edge_features_df['link_centrality'].max():.6f}")
logger.info(f"技术距离范围: {edge_features_df['technical_distance'].min():.6f} - {edge_features_df['technical_distance'].max():.6f}")
logger.info(f"节点同配性范围: {edge_features_df['node_assortativity'].min()} - {edge_features_df['node_assortativity'].max()}")

# 保存特征相关性矩阵
correlation_cols = ['weighted_tie_strength', 'common_neighbors_count', 'link_centrality', 'technical_distance', 'node_assortativity']
correlation_matrix = edge_features_df[correlation_cols].corr()
correlation_filename = f"feature_correlation_{timestamp}.csv"
correlation_path = os.path.join(data_dir, correlation_filename)
correlation_matrix.to_csv(correlation_path)
logger.info(f"特征相关性矩阵已保存到: {correlation_path}")

# 打印前10个最强链接作为示例
logger.info("\n前10个最强链接:")
top_10_strong = keyword_strength_df.head(10)
for idx, row in top_10_strong.iterrows():
    logger.info(f"{row['关键词1']} - {row['关键词2']}: {row['链路强度值']:.4f} ({row['链接强度分类']})")

logger.info("所有计算完成！")