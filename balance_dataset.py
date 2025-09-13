"""
平衡数据集构建脚本
处理Y变量严重不平衡问题（966881个0 vs 30104个1）
采用分层负采样策略，确保时间窗口分布的一致性
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from collections import Counter

# 设置随机种子，确保结果可重复
np.random.seed(42)

# 文件路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_FILE = os.path.join(DATA_DIR, "target_variables_Y20250903_213417.csv")

# 读取数据
print("=" * 60)
print("读取原始数据...")
df = pd.read_csv(INPUT_FILE)
print(f"原始数据规模: {len(df)} 条记录")

# 统计Y值分布
y_counts = df['y'].value_counts()
print(f"\n原始Y值分布:")
print(f"  Y=0: {y_counts.get(0, 0):,} 条")
print(f"  Y=1: {y_counts.get(1, 0):,} 条")
print(f"  不平衡比例: {y_counts.get(0, 0) / y_counts.get(1, 0):.2f}:1")

# 分离正负样本
df_positive = df[df['y'] == 1].copy()
df_negative = df[df['y'] == 0].copy()

print(f"\n正样本数量: {len(df_positive):,}")
print(f"负样本数量: {len(df_negative):,}")

# 采样策略配置
SAMPLING_RATIOS = {
    '1:1': 1,      # 完全平衡
    '1:2': 2,      # 轻度不平衡
    '1:3': 3,      # 中度不平衡
    '1:5': 5       # 保留一定不平衡
}

# 选择采样比例（可根据需要调整）
SELECTED_RATIO = '1:1'  # 负样本是正样本的2倍
ratio_multiplier = SAMPLING_RATIOS[SELECTED_RATIO]

print(f"\n采样策略: {SELECTED_RATIO} (负样本:正样本)")

# 计算需要采样的负样本数量
n_positive = len(df_positive)
n_negative_sample = min(n_positive * ratio_multiplier, len(df_negative))

print(f"目标负样本数量: {n_negative_sample:,}")

# 分层负采样：按时间窗口比例采样
print("\n执行分层负采样...")

# 计算每个时间窗口的正样本分布
window_positive_dist = df_positive.groupby(['window_t_start', 'window_t_end']).size()
window_positive_ratio = window_positive_dist / window_positive_dist.sum()

# 对负样本按时间窗口分组
negative_groups = df_negative.groupby(['window_t_start', 'window_t_end'])

# 按比例从每个时间窗口采样负样本
sampled_negative_dfs = []

for (window_start, window_end), ratio in window_positive_ratio.items():
    # 该时间窗口应采样的负样本数
    n_sample_window = int(n_negative_sample * ratio)
    
    # 获取该时间窗口的负样本
    if (window_start, window_end) in negative_groups.groups:
        window_negative_df = negative_groups.get_group((window_start, window_end))
        
        # 如果该窗口负样本不足，则全部取用
        if len(window_negative_df) <= n_sample_window:
            sampled_df = window_negative_df
        else:
            # 随机采样
            sampled_df = window_negative_df.sample(n=n_sample_window, replace=False)
        
        sampled_negative_dfs.append(sampled_df)
        print(f"  窗口 {window_start}-{window_end}: "
              f"采样 {len(sampled_df):,}/{len(window_negative_df):,} 条负样本")

# 合并采样后的负样本
df_negative_sampled = pd.concat(sampled_negative_dfs, ignore_index=True)

# 合并正负样本，构建平衡数据集
df_balanced = pd.concat([df_positive, df_negative_sampled], ignore_index=True)

# 随机打乱数据
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n平衡后数据集规模: {len(df_balanced):,} 条")
print(f"  Y=0: {len(df_balanced[df_balanced['y']==0]):,} 条")
print(f"  Y=1: {len(df_balanced[df_balanced['y']==1]):,} 条")
print(f"  新比例: {len(df_balanced[df_balanced['y']==0]) / len(df_balanced[df_balanced['y']==1]):.2f}:1")

# 统计各时间窗口的样本分布
print("\n各时间窗口样本分布:")
window_dist = df_balanced.groupby(['window_t_start', 'window_t_end', 'y']).size().unstack(fill_value=0)
print(window_dist)
print("\n各时间窗口正样本比例:")
window_dist['positive_ratio'] = window_dist[1] / (window_dist[0] + window_dist[1])
print(window_dist[['positive_ratio']].apply(lambda x: f"{x.values[0]:.2%}"))

# 构建基本样本数据表
df_final = df_balanced[['window_t_start', 'window_t_end', 
                        'window_t1_start', 'window_t1_end',
                        'node_u', 'node_v', 'y']].copy()

# 添加样本ID
df_final.insert(0, 'sample_id', range(1, len(df_final) + 1))

# 添加时间窗口标签
df_final['window_label'] = df_final.apply(
    lambda x: f"W{x['window_t_start']}-{x['window_t_end']}", axis=1
)

# 生成输出文件名
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"RES01_balanced_dataset_{SELECTED_RATIO.replace(':', '_')}_{timestamp}.csv"
output_path = os.path.join(DATA_DIR, output_filename)

# 保存数据
df_final.to_csv(output_path, index=False, encoding='utf-8-sig')

print("\n" + "=" * 60)
print(f"✅ 数据集已保存: {output_filename}")
print(f"   路径: {output_path}")
print(f"   记录数: {len(df_final):,}")
print("=" * 60)

# 生成数据集统计报告
report_filename = f"RES01_dataset_report_{timestamp}.txt"
report_path = os.path.join(DATA_DIR, report_filename)

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("平衡数据集构建报告\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"采样策略: {SELECTED_RATIO} (负样本:正样本)\n\n")
    
    f.write("原始数据分布:\n")
    f.write(f"  Y=0: {y_counts.get(0, 0):,} 条\n")
    f.write(f"  Y=1: {y_counts.get(1, 0):,} 条\n")
    f.write(f"  不平衡比例: {y_counts.get(0, 0) / y_counts.get(1, 0):.2f}:1\n\n")
    
    f.write("平衡后数据分布:\n")
    f.write(f"  总样本数: {len(df_balanced):,} 条\n")
    f.write(f"  Y=0: {len(df_balanced[df_balanced['y']==0]):,} 条\n")
    f.write(f"  Y=1: {len(df_balanced[df_balanced['y']==1]):,} 条\n")
    f.write(f"  新比例: {len(df_balanced[df_balanced['y']==0]) / len(df_balanced[df_balanced['y']==1]):.2f}:1\n\n")
    
    f.write("各时间窗口分布:\n")
    f.write(str(window_dist))
    f.write("\n\n数据字段说明:\n")
    f.write("  sample_id: 样本唯一标识\n")
    f.write("  window_t_start: t时刻窗口起始年\n")
    f.write("  window_t_end: t时刻窗口结束年\n")
    f.write("  window_t1_start: t+1时刻窗口起始年\n")
    f.write("  window_t1_end: t+1时刻窗口结束年\n")
    f.write("  node_u: 关键词节点1\n")
    f.write("  node_v: 关键词节点2\n")
    f.write("  y: 目标变量 (0=保持弱链接, 1=转变为强链接)\n")
    f.write("  window_label: 时间窗口标签\n")

print(f"📊 统计报告已生成: {report_filename}")

# 可选：生成额外的采样比例数据集
print("\n是否生成其他采样比例的数据集? (用于对比实验)")
print("可选比例: 1:1, 1:3, 1:5")
print("(注：此步骤可选，按需启用)")

# 如需生成多个比例的数据集，可取消下面的注释
"""
for ratio_name, ratio_value in SAMPLING_RATIOS.items():
    if ratio_name != SELECTED_RATIO:  # 跳过已生成的比例
        n_neg_sample = min(n_positive * ratio_value, len(df_negative))
        # ... (重复上述采样逻辑)
"""

print("\n✅ 所有任务完成！")
