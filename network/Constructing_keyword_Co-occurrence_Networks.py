import pandas as pd
import ast
from collections import defaultdict
from datetime import datetime
import os
from itertools import combinations
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义输入文件路径
input_file = "data/Keywords_Extracted_20250830_144617.csv"

# 读取CSV文件
df = pd.read_csv(input_file)

# 解析keywords列，将字符串转换为列表
def parse_keywords(keyword_str):
    try:
        return ast.literal_eval(keyword_str)
    except Exception as e:
        logger.warning(f"解析关键词失败: {keyword_str}, 错误: {e}")
        return []

df['keywords'] = df['keywords'].apply(parse_keywords)

# 移除空关键词列表
original_count = len(df)
df = df[df['keywords'].apply(len) > 0]
if len(df) < original_count:
    logger.info(f"移除了 {original_count - len(df)} 行空关键词数据")

# 构建共现网络
co_occurrence = defaultdict(int)

for keywords_list in df['keywords']:
    unique_keywords = list(set(keywords_list))
    if len(unique_keywords) < 2:
        continue  # 跳过只有一个关键词的专利
        
    for pair in combinations(unique_keywords, 2):
        sorted_pair = tuple(sorted(pair))
        co_occurrence[sorted_pair] += 1

# 可选：设置最小共现阈值
min_co_occurrence = 1  # 可根据需要调整
co_occurrence = {k: v for k, v in co_occurrence.items() if v >= min_co_occurrence}

# 将共现字典转换为DataFrame
edges = []
for pair, count in co_occurrence.items():
    edges.append({
        'source': pair[0],
        'target': pair[1],
        'weight': count
    })

co_occurrence_df = pd.DataFrame(edges)

# 生成时间戳
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"keyword_co_network_{timestamp}.csv"
output_path = os.path.join("data", output_filename)

# 确保data目录存在
os.makedirs("data", exist_ok=True)

# 保存到CSV文件
co_occurrence_df.to_csv(output_path, index=False)

logger.info(f"共现网络已保存到: {output_path}")
logger.info(f"共生成 {len(co_occurrence_df)} 条边")