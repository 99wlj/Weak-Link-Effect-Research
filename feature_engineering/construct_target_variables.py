import pandas as pd
import datetime
import os

# 输入数据，待修改
base_dir = r"F:\WLJ\Weak-Link-Effect-Research"
data_dir = os.path.join(base_dir, "data")
csv_file = os.path.join(data_dir, "Keyword_LinkStrength_ByWindow_20250903_212630.csv")

df = pd.read_csv(csv_file)

# 构建字典：按 (window_start, window_end) 分组，存储每条边的特征
time_dict = {}
for (ws, we), group in df.groupby(["window_start", "window_end"]):
    edges = {}
    for _, row in group.iterrows():
        edges[(row["node_u"], row["node_v"])] = row.to_dict()
    time_dict[(ws, we)] = edges

results = []

# 遍历每个时间窗口
sorted_windows = sorted(time_dict.keys())
for i in range(len(sorted_windows)-1):
    t = sorted_windows[i]
    t1 = sorted_windows[i+1]
    edges_t = time_dict[t]
    edges_t1 = time_dict[t1]

    for (u,v), row_t in edges_t.items():
        if row_t["strength_type"] == "weak":
            y = 0
            if (u,v) in edges_t1 and edges_t1[(u,v)]["strength_type"] == "strong":
                y = 1

            result_row = {
                "window_t_start": t[0],
                "window_t_end": t[1],
                "window_t1_start": t1[0],
                "window_t1_end": t1[1],
                "node_u": u,
                "node_v": v,
                "y": y,
            }
            results.append(result_row)

# 转换为 DataFrame
df_y = pd.DataFrame(results)

# 生成时间戳
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 确保 data 目录存在
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# 输出 CSV 文件
output_file = os.path.join(output_dir, f"target_variables_Y{timestamp}.csv")
df_y.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"✅ 文件已保存: {output_file}")
