"""
从Keywords_Processed文件生成关键词共现链接文件
基于专利中关键词的共现关系构建网络边数据
"""

import pandas as pd
import numpy as np
from itertools import combinations
from collections import Counter
from datetime import datetime
import ast
from tqdm import tqdm
import os

# 设置随机种子
np.random.seed(42)

# 配置
DATA_DIR = 'F:/WLJ/Weak-Link-Effect-Research/data'

# 输入文件路径
KEYWORDS_FILE = os.path.join(DATA_DIR, 'Keywords_Processed_20250903_211104.csv')

print("=" * 80)
print("生成关键词共现链接文件")
print("=" * 80)

# 1. 读取关键词数据
print("\n[1] 读取关键词数据...")
df_keywords = pd.read_csv(KEYWORDS_FILE)
print(f"  总记录数: {len(df_keywords):,}")
print(f"  年份范围: {df_keywords['earliest_publn_year'].min()} - {df_keywords['earliest_publn_year'].max()}")

# 2. 解析关键词并生成共现关系
print("\n[2] 提取关键词共现关系...")

co_occurrence_records = []
keyword_stats = Counter()
year_stats = Counter()

for idx, row in tqdm(df_keywords.iterrows(), total=len(df_keywords), desc="处理专利"):
    # 获取基本信息
    appln_id = row['appln_id']
    year = row['earliest_publn_year']
    
    # 解析关键词列表
    try:
        # 尝试用ast.literal_eval解析（如果是字符串形式的列表）
        keywords_str = row['keywords']
        if pd.isna(keywords_str) or keywords_str == '[]':
            continue
        keywords = ast.literal_eval(keywords_str)
    except:
        # 如果解析失败，尝试其他方式
        try:
            keywords = row['keywords'].split(',') if isinstance(row['keywords'], str) else []
            # 清理关键词（去除多余空格）
            keywords = [kw.strip() for kw in keywords if kw.strip()]
        except:
            continue
    
    # 跳过空列表或单个关键词的情况
    if len(keywords) < 2:
        continue
    
    # 统计关键词出现次数
    for kw in keywords:
        keyword_stats[kw] += 1
    
    # 统计年份
    year_stats[year] += 1
    
    # 生成关键词对（组合）- 确保顺序一致性
    for kw1, kw2 in combinations(sorted(keywords), 2):
        co_occurrence_records.append({
            'appln_id': appln_id,
            'year': year,
            'keyword1': kw1,
            'keyword2': kw2
        })

print(f"\n  找到 {len(co_occurrence_records):,} 个共现关系")
print(f"  涉及 {len(keyword_stats):,} 个不同关键词")

# 3. 转换为DataFrame并统计
print("\n[3] 统计共现频次...")
df_cooc = pd.DataFrame(co_occurrence_records)

# 按年份和关键词对分组，统计共现次数
df_links = df_cooc.groupby(['year', 'keyword1', 'keyword2']).agg({
    'appln_id': 'count'  # 统计共现次数
}).reset_index()

# 重命名列
df_links.rename(columns={'appln_id': 'co_occurrences'}, inplace=True)

# 添加边的ID
df_links.insert(0, 'edge_id', range(1, len(df_links) + 1))

# 4. 计算额外的统计信息
print("\n[4] 计算边的统计信息...")

# 为每条边计算关键词的总出现次数（用于后续分析）
df_links['keyword1_freq'] = df_links['keyword1'].map(keyword_stats)
df_links['keyword2_freq'] = df_links['keyword2'].map(keyword_stats)

# 计算边的权重（可以基于共现次数和关键词频率）
# 使用Jaccard相似度的思想
df_links['edge_weight'] = df_links['co_occurrences'] / (
    df_links['keyword1_freq'] + df_links['keyword2_freq'] - df_links['co_occurrences']
)

# 5. 数据清理和排序
print("\n[5] 数据清理和排序...")

# 按年份和共现次数排序
df_links = df_links.sort_values(['year', 'co_occurrences'], ascending=[True, False])

# 选择要保存的列
columns_to_save = [
    'edge_id', 
    'year', 
    'keyword1', 
    'keyword2', 
    'co_occurrences',
    'edge_weight',
    'keyword1_freq',
    'keyword2_freq'
]

df_final = df_links[columns_to_save].copy()

# 6. 统计信息输出
print("\n" + "=" * 80)
print("共现链接统计:")
print("=" * 80)
print(f"  总边数: {len(df_final):,}")
print(f"  年份范围: {df_final['year'].min()} - {df_final['year'].max()}")
print(f"  平均共现次数: {df_final['co_occurrences'].mean():.2f}")
print(f"  最大共现次数: {df_final['co_occurrences'].max()}")
print(f"  最小共现次数: {df_final['co_occurrences'].min()}")

# 各年份边数统计
print("\n各年份统计:")
year_edge_stats = df_final.groupby('year').agg({
    'edge_id': 'count',
    'co_occurrences': ['sum', 'mean', 'max']
}).round(2)
year_edge_stats.columns = ['边数', '总共现次数', '平均共现', '最大共现']
print(year_edge_stats.head(10))

# Top关键词对
print("\nTop 10 最频繁的关键词对:")
top_pairs = df_final.nlargest(10, 'co_occurrences')[
    ['keyword1', 'keyword2', 'co_occurrences', 'year']
]
for idx, row in top_pairs.iterrows():
    print(f"  {row['keyword1']} <-> {row['keyword2']}: {row['co_occurrences']} 次 ({row['year']}年)")

# 7. 保存文件
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"keyword_co_occurrence_links_{timestamp}.csv"
output_path = os.path.join(DATA_DIR, output_filename)

df_final.to_csv(output_path, index=False, encoding='utf-8-sig')

print("\n" + "=" * 80)
print(f"✅ 关键词共现链接文件已生成!")
print(f"   文件名: {output_filename}")
print(f"   路径: {output_path}")
print(f"   总边数: {len(df_final):,}")
print("=" * 80)

# 8. 生成详细报告
report_filename = f"keyword_co_occurrence_report_{timestamp}.txt"
report_path = os.path.join(DATA_DIR, report_filename)

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("关键词共现链接生成报告\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"源文件: {KEYWORDS_FILE}\n\n")
    
    f.write("数据概览:\n")
    f.write(f"  输入专利数: {len(df_keywords):,}\n")
    f.write(f"  生成共现对数: {len(co_occurrence_records):,}\n")
    f.write(f"  聚合后边数: {len(df_final):,}\n")
    f.write(f"  涉及关键词数: {len(keyword_stats):,}\n\n")
    
    f.write("时间分布:\n")
    f.write(f"  年份范围: {df_final['year'].min()} - {df_final['year'].max()}\n")
    f.write(f"  年份数量: {df_final['year'].nunique()}\n\n")
    
    f.write("共现统计:\n")
    f.write(f"  平均共现次数: {df_final['co_occurrences'].mean():.2f}\n")
    f.write(f"  中位数: {df_final['co_occurrences'].median():.0f}\n")
    f.write(f"  标准差: {df_final['co_occurrences'].std():.2f}\n")
    f.write(f"  最小值: {df_final['co_occurrences'].min()}\n")
    f.write(f"  最大值: {df_final['co_occurrences'].max()}\n\n")
    
    f.write("边权重分布:\n")
    weight_stats = df_final['edge_weight'].describe()
    f.write(str(weight_stats))
    f.write("\n\n")
    
    f.write("各年份详细统计:\n")
    f.write(str(year_edge_stats))
    f.write("\n\n")
    
    f.write("Top 20 高频关键词:\n")
    for kw, count in keyword_stats.most_common(20):
        f.write(f"  {kw}: {count} 次\n")
    f.write("\n")
    
    f.write("Top 20 高频共现对:\n")
    top20_pairs = df_final.nlargest(20, 'co_occurrences')
    for idx, row in top20_pairs.iterrows():
        f.write(f"  {row['keyword1']} <-> {row['keyword2']}: ")
        f.write(f"{row['co_occurrences']} 次 (权重: {row['edge_weight']:.4f})\n")
    
    f.write("\n数据字段说明:\n")
    f.write("  edge_id: 边的唯一标识符\n")
    f.write("  year: 共现发生的年份\n")
    f.write("  keyword1, keyword2: 共现的两个关键词\n")
    f.write("  co_occurrences: 共现次数（在多少个专利中同时出现）\n")
    f.write("  edge_weight: 边权重（基于Jaccard相似度）\n")
    f.write("  keyword1_freq, keyword2_freq: 各关键词的总出现频率\n")

print(f"📊 详细报告已生成: {report_filename}")

# 9. 生成可视化（可选）
print("\n生成共现分布图...")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 共现次数分布
axes[0, 0].hist(df_final['co_occurrences'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('共现次数')
axes[0, 0].set_ylabel('频数')
axes[0, 0].set_title('共现次数分布')
axes[0, 0].set_yscale('log')

# 2. 边权重分布
axes[0, 1].hist(df_final['edge_weight'], bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[0, 1].set_xlabel('边权重')
axes[0, 1].set_ylabel('频数')
axes[0, 1].set_title('边权重分布')

# 3. 年份边数变化
year_counts = df_final.groupby('year')['edge_id'].count()
axes[1, 0].plot(year_counts.index, year_counts.values, marker='o')
axes[1, 0].set_xlabel('年份')
axes[1, 0].set_ylabel('边数')
axes[1, 0].set_title('各年份边数变化')
axes[1, 0].grid(True, alpha=0.3)

# 4. 共现次数年度变化
year_cooc = df_final.groupby('year')['co_occurrences'].agg(['mean', 'max'])
axes[1, 1].plot(year_cooc.index, year_cooc['mean'], marker='o', label='平均值')
axes[1, 1].plot(year_cooc.index, year_cooc['max'], marker='s', label='最大值', alpha=0.7)
axes[1, 1].set_xlabel('年份')
axes[1, 1].set_ylabel('共现次数')
axes[1, 1].set_title('共现次数年度变化')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plot_filename = f"keyword_co_occurrence_stats_{timestamp}.png"
plt.savefig(os.path.join(DATA_DIR, plot_filename), dpi=150)
print(f"📈 统计图已保存: {plot_filename}")

print("\n✅ 所有任务完成！")
