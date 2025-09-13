"""
基于专利数的简化采样策略
根据节点涉及的专利数筛选高质量样本
目标：每个时间窗口保留3000-5000个样本
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from collections import Counter
from tqdm import tqdm
import ast

# 设置随机种子
np.random.seed(42)

# 配置
DATA_DIR = 'F:/WLJ/Weak-Link-Effect-Research/data'
TARGET_SAMPLES_PER_WINDOW = 3000  # 每个时间窗口的目标样本数

# 文件路径
INPUT_FILE = os.path.join(DATA_DIR, "target_variables_Y20250903_213417.csv")
LINKS_FILE = os.path.join(DATA_DIR, "keyword_co_occurrence_links_20250913_235649.csv")
KEYWORDS_FILE = os.path.join(DATA_DIR, "Keywords_Processed_20250903_211104.csv")

print("=" * 80)
print("基于专利数的简化智能采样")
print("=" * 80)

# 1. 读取数据
print("\n[1] 读取数据...")
df = pd.read_csv(INPUT_FILE)
df_keywords = pd.read_csv(KEYWORDS_FILE)

print(f"  原始样本数: {len(df):,}")
print(f"  Y=0: {(df['y']==0).sum():,}")
print(f"  Y=1: {(df['y']==1).sum():,}")

# 2. 计算节点的专利数（简化的重要性指标）
print("\n[2] 计算节点专利数...")

# 获取所有时间窗口
time_windows = df[['window_t_start', 'window_t_end']].drop_duplicates()
node_patent_counts = {}

for _, (w_start, w_end) in tqdm(time_windows.iterrows(), total=len(time_windows), desc="处理时间窗口"):
    # 获取该时间窗口的关键词数据
    window_keywords = df_keywords[
        (df_keywords['earliest_publn_year'] >= w_start) & 
        (df_keywords['earliest_publn_year'] <= w_end)
    ]
    
    # 统计每个关键词涉及的专利数
    keyword_counts = Counter()
    
    for _, row in window_keywords.iterrows():
        try:
            keywords = ast.literal_eval(row['keywords'])
            for keyword in keywords:
                keyword_counts[keyword] += 1
        except:
            continue
    
    # 保存该窗口的关键词专利数
    for keyword, count in keyword_counts.items():
        node_patent_counts[(w_start, w_end, keyword)] = count
    
    print(f"  窗口 {w_start}-{w_end}: {len(keyword_counts)} 个关键词, "
          f"平均专利数: {np.mean(list(keyword_counts.values())):.1f}")

# 3. 计算边的重要性（基于专利数）
print("\n[3] 计算边的重要性...")

def get_edge_importance(node_u, node_v, w_start, w_end):
    """
    计算边的重要性：两个端点专利数的总和
    专利数越多，说明这条边涉及的研究越多，越重要
    """
    count_u = node_patent_counts.get((w_start, w_end, node_u), 0)
    count_v = node_patent_counts.get((w_start, w_end, node_v), 0)
    # 使用总和而不是几何平均，确保专利数多的边得分更高
    return count_u + count_v

# 为每个样本计算边重要性
print("  计算所有边的重要性得分...")
df['edge_importance'] = df.apply(
    lambda x: get_edge_importance(x['node_u'], x['node_v'], 
                                 x['window_t_start'], x['window_t_end']),
    axis=1
)

print(f"\n  边重要性统计:")
print(f"    均值: {df['edge_importance'].mean():.2f}")
print(f"    中位数: {df['edge_importance'].median():.0f}")
print(f"    最大值: {df['edge_importance'].max():.0f}")
print(f"    最小值: {df['edge_importance'].min():.0f}")

# 统计有多少边没有专利数（重要性为0）
zero_importance = (df['edge_importance'] == 0).sum()
if zero_importance > 0:
    print(f"    ⚠️ 警告: {zero_importance:,} 条边的重要性为0（无专利数据）")

# 4. 智能采样
print("\n[4] 执行基于专利数的智能采样...")

sampled_dfs = []
sampling_stats = []

for _, (w_start, w_end) in time_windows.iterrows():
    print(f"\n  窗口 {w_start}-{w_end}:")
    
    # 获取该窗口的样本
    window_df = df[(df['window_t_start'] == w_start) & 
                   (df['window_t_end'] == w_end)].copy()
    
    if len(window_df) == 0:
        continue
    
    # 分离正负样本
    positive_df = window_df[window_df['y'] == 1]
    negative_df = window_df[window_df['y'] == 0]
    
    print(f"    原始: Y=1:{len(positive_df):,}, Y=0:{len(negative_df):,}")
    
    # 统计重要性大于0的样本
    pos_with_importance = (positive_df['edge_importance'] > 0).sum()
    neg_with_importance = (negative_df['edge_importance'] > 0).sum()
    print(f"    有专利数据: Y=1:{pos_with_importance:,}, Y=0:{neg_with_importance:,}")
    
    # === 正样本采样策略 ===
    # 优先保留专利数多的正样本
    positive_df_sorted = positive_df.sort_values('edge_importance', ascending=False)
    
    # 确定正样本数量
    n_positive_target = min(
        int(TARGET_SAMPLES_PER_WINDOW * 0.4),  # 正样本占40%
        int(len(positive_df) * 0.8),           # 或保留80%的正样本
        len(positive_df)                       # 不超过总数
    )
    
    if n_positive_target > 0:
        # 90%按专利数选择（保留高价值样本），10%随机（保证多样性）
        n_by_patents = int(n_positive_target * 0.9)
        n_random = n_positive_target - n_by_patents
        
        # 选择专利数最多的样本
        positive_important = positive_df_sorted.head(n_by_patents)
        
        # 从剩余样本中随机选择
        remaining = positive_df_sorted.iloc[n_by_patents:]
        if len(remaining) > 0 and n_random > 0:
            positive_random = remaining.sample(n=min(n_random, len(remaining)), random_state=42)
            positive_selected = pd.concat([positive_important, positive_random])
        else:
            positive_selected = positive_important
    else:
        positive_selected = pd.DataFrame()
    
    # === 负样本采样策略 ===
    n_negative_target = min(
        TARGET_SAMPLES_PER_WINDOW - len(positive_selected),  # 补齐到目标数量
        int(len(positive_selected) * 1.5),                   # 或正样本的1.5倍
        len(negative_df)                                     # 不超过总数
    )
    
    if n_negative_target > 0 and len(negative_df) > 0:
        # 负样本也按专利数分层，确保覆盖不同重要性级别
        negative_df_sorted = negative_df.sort_values('edge_importance', ascending=False)
        
        # 过滤掉重要性为0的样本（如果有足够的非零样本）
        non_zero_negative = negative_df_sorted[negative_df_sorted['edge_importance'] > 0]
        
        if len(non_zero_negative) >= n_negative_target:
            # 有足够的非零样本，只从中选择
            negative_df_sorted = non_zero_negative
        
        # 分三层：高、中、低专利数
        n_samples = len(negative_df_sorted)
        
        if n_samples >= 3:
            # 计算每层的边界
            n_high = n_samples // 3
            n_mid = n_samples // 3
            
            high_patent = negative_df_sorted.iloc[:n_high]
            mid_patent = negative_df_sorted.iloc[n_high:n_high+n_mid]
            low_patent = negative_df_sorted.iloc[n_high+n_mid:]
            
            # 采样比例：50% 高专利数，35% 中专利数，15% 低专利数
            # 优先选择专利数多的样本
            n_high_sample = min(int(n_negative_target * 0.5), len(high_patent))
            n_mid_sample = min(int(n_negative_target * 0.35), len(mid_patent))
            n_low_sample = min(n_negative_target - n_high_sample - n_mid_sample, len(low_patent))
            
            negative_selected = pd.concat([
                high_patent.sample(n=n_high_sample, random_state=42) if n_high_sample > 0 else pd.DataFrame(),
                mid_patent.sample(n=n_mid_sample, random_state=42) if n_mid_sample > 0 else pd.DataFrame(),
                low_patent.sample(n=n_low_sample, random_state=42) if n_low_sample > 0 else pd.DataFrame()
            ])
        else:
            # 样本太少，直接选择
            negative_selected = negative_df_sorted.head(n_negative_target)
    else:
        negative_selected = pd.DataFrame()
    
    # 合并该窗口的样本
    window_sampled = pd.concat([positive_selected, negative_selected])
    sampled_dfs.append(window_sampled)
    
    # 记录采样统计
    sampling_stats.append({
        'window': f"{w_start}-{w_end}",
        'original_pos': len(positive_df),
        'original_neg': len(negative_df),
        'sampled_pos': len(positive_selected),
        'sampled_neg': len(negative_selected),
        'avg_importance': window_sampled['edge_importance'].mean() if len(window_sampled) > 0 else 0
    })
    
    print(f"    采样后: Y=1:{len(positive_selected):,}, Y=0:{len(negative_selected):,}")
    print(f"    平均专利数: {window_sampled['edge_importance'].mean():.1f}")

# 5. 合并所有窗口的样本
df_sampled = pd.concat(sampled_dfs, ignore_index=True)

# 随机打乱
df_sampled = df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)

print("\n" + "=" * 80)
print("采样结果汇总:")
print(f"  总样本数: {len(df_sampled):,}")
print(f"  Y=0: {(df_sampled['y']==0).sum():,}")
print(f"  Y=1: {(df_sampled['y']==1).sum():,}")
print(f"  比例: {(df_sampled['y']==0).sum() / (df_sampled['y']==1).sum():.2f}:1")
print(f"  平均边专利数: {df_sampled['edge_importance'].mean():.1f}")
print(f"  中位数边专利数: {df_sampled['edge_importance'].median():.0f}")

# 比较采样前后的专利数分布
print(f"\n专利数提升:")
print(f"  采样前平均: {df['edge_importance'].mean():.1f}")
print(f"  采样后平均: {df_sampled['edge_importance'].mean():.1f}")
print(f"  提升比例: {(df_sampled['edge_importance'].mean() / df['edge_importance'].mean() - 1) * 100:.1f}%")

# 各窗口分布
print("\n各时间窗口分布:")
window_dist = df_sampled.groupby(['window_t_start', 'window_t_end', 'y']).size().unstack(fill_value=0)
window_dist['total'] = window_dist[0] + window_dist[1]
window_dist['positive_ratio'] = window_dist[1] / window_dist['total']
window_dist['avg_importance'] = df_sampled.groupby(['window_t_start', 'window_t_end'])['edge_importance'].mean()
print(window_dist)

# 6. 构建最终数据集
df_final = df_sampled[['window_t_start', 'window_t_end', 
                       'window_t1_start', 'window_t1_end',
                       'node_u', 'node_v', 'y', 'edge_importance']].copy()

# 添加样本ID
df_final.insert(0, 'sample_id', range(1, len(df_final) + 1))

# 添加窗口标签
df_final['window_label'] = df_final.apply(
    lambda x: f"W{x['window_t_start']}-{x['window_t_end']}", axis=1
)

# 添加专利数等级（便于分析）
df_final['patent_level'] = pd.cut(df_final['edge_importance'], 
                                  bins=[0, 10, 50, 100, float('inf')],
                                  labels=['低(≤10)', '中(11-50)', '高(51-100)', '很高(>100)'])

# 7. 保存结果
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"RES01_patent_based_sampled_{timestamp}.csv"
output_path = os.path.join(DATA_DIR, output_filename)

df_final.to_csv(output_path, index=False, encoding='utf-8-sig')

print("\n" + "=" * 80)
print(f"✅ 基于专利数的采样数据集已保存!")
print(f"   文件名: {output_filename}")
print(f"   路径: {output_path}")
print(f"   总样本数: {len(df_final):,}")
print("=" * 80)

# 8. 生成详细报告
report_filename = f"RES01_patent_sampling_report_{timestamp}.txt"
report_path = os.path.join(DATA_DIR, report_filename)

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("基于专利数的智能采样报告\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"采样策略: 基于边涉及的专利数进行重要性排序\n")
    f.write(f"目标样本数/窗口: {TARGET_SAMPLES_PER_WINDOW}\n\n")
    
    f.write("原始数据分布:\n")
    f.write(f"  总样本: {len(df):,}\n")
    f.write(f"  Y=0: {(df['y']==0).sum():,}\n")
    f.write(f"  Y=1: {(df['y']==1).sum():,}\n")
    f.write(f"  不平衡比例: {(df['y']==0).sum() / (df['y']==1).sum():.2f}:1\n")
    f.write(f"  平均边专利数: {df['edge_importance'].mean():.1f}\n\n")
    
    f.write("采样后数据分布:\n")
    f.write(f"  总样本: {len(df_sampled):,}\n")
    f.write(f"  Y=0: {(df_sampled['y']==0).sum():,}\n")
    f.write(f"  Y=1: {(df_sampled['y']==1).sum():,}\n")
    f.write(f"  新比例: {(df_sampled['y']==0).sum() / (df_sampled['y']==1).sum():.2f}:1\n")
    f.write(f"  平均边专利数: {df_sampled['edge_importance'].mean():.1f}\n\n")
    
    f.write("专利数提升效果:\n")
    f.write(f"  原始平均专利数: {df['edge_importance'].mean():.1f}\n")
    f.write(f"  采样后平均专利数: {df_sampled['edge_importance'].mean():.1f}\n")
    f.write(f"  提升比例: {(df_sampled['edge_importance'].mean() / df['edge_importance'].mean() - 1) * 100:.1f}%\n\n")
    
    f.write("数据压缩率:\n")
    f.write(f"  压缩比: {len(df) / len(df_sampled):.1f}:1\n")
    f.write(f"  保留比例: {len(df_sampled) / len(df) * 100:.2f}%\n\n")
    
    f.write("专利数等级分布（采样后）:\n")
    patent_level_dist = df_final['patent_level'].value_counts()
    for level, count in patent_level_dist.items():
        f.write(f"  {level}: {count:,} ({count/len(df_final)*100:.1f}%)\n")
    f.write("\n")
    
    f.write("各时间窗口采样统计:\n")
    f.write(f"{'窗口':^12} | {'原Y=1':>8} | {'原Y=0':>8} | {'采Y=1':>8} | {'采Y=0':>8} | {'平均专利数':>10}\n")
    f.write("-" * 70 + "\n")
    for stat in sampling_stats:
        f.write(f"{stat['window']:^12} | {stat['original_pos']:>8,} | {stat['original_neg']:>8,} | ")
        f.write(f"{stat['sampled_pos']:>8,} | {stat['sampled_neg']:>8,} | {stat['avg_importance']:>10.1f}\n")
    
    f.write("\n边重要性计算方法:\n")
    f.write("  边重要性 = node_u的专利数 + node_v的专利数\n")
    f.write("  专利数越多，说明该边涉及的研究越活跃，重要性越高\n\n")
    
    f.write("采样策略:\n")
    f.write("  正样本(Y=1):\n")
    f.write("    - 保留专利数最多的80%样本\n")
    f.write("    - 90%按专利数排序选择\n")
    f.write("    - 10%随机选择（保证多样性）\n")
    f.write("  负样本(Y=0):\n")
    f.write("    - 分层采样：50%高专利数、35%中专利数、15%低专利数\n")
    f.write("    - 优先选择专利数>0的样本\n\n")

print(f"📊 详细报告已生成: {report_filename}")

# 9. 可视化
print("\n生成可视化图表...")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 专利数分布对比
axes[0, 0].hist([df['edge_importance'], df_sampled['edge_importance']], 
                bins=30, alpha=0.7, label=['原始', '采样后'])
axes[0, 0].set_xlabel('边专利数')
axes[0, 0].set_ylabel('频数')
axes[0, 0].set_title('专利数分布对比')
axes[0, 0].legend()
axes[0, 0].set_yscale('log')

# 2. Y=0和Y=1的专利数分布（采样后）
axes[0, 1].boxplot([df_sampled[df_sampled['y']==0]['edge_importance'],
                    df_sampled[df_sampled['y']==1]['edge_importance']],
                   labels=['Y=0', 'Y=1'])
axes[0, 1].set_ylabel('边专利数')
axes[0, 1].set_title('不同类别的专利数分布（采样后）')
axes[0, 1].grid(True, alpha=0.3)

# 3. 各窗口的平均专利数
window_importance = df_sampled.groupby(['window_t_start', 'window_t_end'])['edge_importance'].mean()
x_labels = [f"{idx[0]}-{idx[1]}" for idx in window_importance.index]
axes[1, 0].bar(range(len(window_importance)), window_importance.values)
axes[1, 0].set_xlabel('时间窗口')
axes[1, 0].set_ylabel('平均专利数')
axes[1, 0].set_title('各时间窗口的平均专利数')
axes[1, 0].set_xticks(range(len(x_labels)))
axes[1, 0].set_xticklabels(x_labels, rotation=45)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 4. 采样比例vs专利数
sampling_df = pd.DataFrame(sampling_stats)
axes[1, 1].scatter(sampling_df['avg_importance'], 
                  sampling_df['sampled_pos'] / (sampling_df['sampled_pos'] + sampling_df['sampled_neg']),
                  s=100, alpha=0.6)
axes[1, 1].set_xlabel('平均专利数')
axes[1, 1].set_ylabel('正样本比例')
axes[1, 1].set_title('专利数与正样本比例的关系')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plot_filename = f"RES01_patent_sampling_viz_{timestamp}.png"
plt.savefig(os.path.join(DATA_DIR, plot_filename), dpi=150)
print(f"📈 可视化图表已保存: {plot_filename}")

print("\n✅ 所有任务完成！")
print(f"   最终数据集: {len(df_final):,} 个样本")
print(f"   平均专利数: {df_final['edge_importance'].mean():.1f}")
print(f"   相比原始数据，专利数提升了 {(df_final['edge_importance'].mean() / df['edge_importance'].mean() - 1) * 100:.1f}%")
