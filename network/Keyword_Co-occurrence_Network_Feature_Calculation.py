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
    
    for idx, row in edges_df.iterrows():
        source = row['source']
        target = row['target']
        weight = row['weight']
        
        # 1. 链路强度 - 直接使用共现次数（weight）
        link_strength = weight
        
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
        
        features.append({
            'source': source,
            'target': target,
            'weight': weight,
            'link_strength': link_strength,
            'link_centrality': link_centrality,
            'technical_distance': technical_distance,
            'node_assortativity': node_assortativity,
            'eigen_distance': eigen_distance,  # 分解特征，便于分析
            'pagerank_distance': pagerank_distance,
            'clustering_distance': clustering_distance
        })
    
    return pd.DataFrame(features)

# 计算边特征
logger.info("开始计算边特征...")
edge_features_df = calculate_edge_features(G, edges_df)
logger.info("边特征计算完成")

# 生成时间戳
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"keyword_co_network_with_features_{timestamp}.csv"
# 修改保存路径到data文件夹
output_path = os.path.join(data_dir, output_filename)

# 保存到CSV文件
edge_features_df.to_csv(output_path, index=False)

logger.info(f"带特征的共现网络已保存到: {output_path}")
logger.info(f"共处理 {len(edge_features_df)} 条边")

# 打印特征统计信息
logger.info("\n特征统计信息:")
logger.info(f"链路强度范围: {edge_features_df['link_strength'].min()} - {edge_features_df['link_strength'].max()}")
logger.info(f"链路中心性范围: {edge_features_df['link_centrality'].min():.6f} - {edge_features_df['link_centrality'].max():.6f}")
logger.info(f"技术距离范围: {edge_features_df['technical_distance'].min():.6f} - {edge_features_df['technical_distance'].max():.6f}")
logger.info(f"节点同配性范围: {edge_features_df['node_assortativity'].min()} - {edge_features_df['node_assortativity'].max()}")

# 保存特征相关性矩阵（可选）
correlation_matrix = edge_features_df[['link_strength', 'link_centrality', 'technical_distance', 'node_assortativity']].corr()
correlation_filename = f"feature_correlation_{timestamp}.csv"
# 修改保存路径到data文件夹
correlation_path = os.path.join(data_dir, correlation_filename)
correlation_matrix.to_csv(correlation_path)
logger.info(f"特征相关性矩阵已保存到: {correlation_path}")