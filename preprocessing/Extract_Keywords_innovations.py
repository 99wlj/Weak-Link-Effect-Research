import pandas as pd
import ast
import re
from datetime import datetime
import os

# 获取项目根目录（当前脚本的上级目录）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据文件路径（在 data 文件夹里）
file_path = os.path.join(BASE_DIR, "data", "my_raw_data.csv")

# 输出文件路径
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
keywords_output_path = os.path.join(BASE_DIR, "data", f"Keywords_Extracted_{timestamp}.csv")
innovations_output_path = os.path.join(BASE_DIR, "data", f"Innovations_Extracted_{timestamp}.csv")

# 改进的关键词提取函数
def extract_keywords(result_str):
    if pd.isna(result_str):
        return []
    
    # 使用正则表达式匹配方括号内的内容
    keyword_match = re.search(r'Keywords:\s*\[(.*?)\]', result_str)
    if keyword_match:
        keywords_str = keyword_match.group(1)
        # 分割关键词并去除前后空格
        keywords = [k.strip() for k in keywords_str.split(',')]
        return keywords
    return []

# 创新点提取函数
def extract_innovations(result_str):
    if pd.isna(result_str):
        return []
    
    # 查找"Innovation:"部分
    innovation_match = re.search(r'Innovation:(.*?)(?=Keywords:|$)', result_str, re.DOTALL)
    if innovation_match:
        innovation_text = innovation_match.group(1).strip()
        
        # 提取编号的创新点
        innovations = re.findall(r'\d+\.\s*(.*?)(?=\d+\.|$)', innovation_text, re.DOTALL)
        if innovations:
            return [innovation.strip() for innovation in innovations if innovation.strip()]
        
        # 如果没有编号，尝试按行分割
        lines = innovation_text.split('\n')
        return [line.strip() for line in lines if line.strip() and not line.strip().startswith('Keywords:')]
    
    return []

# 读取CSV
df = pd.read_csv(file_path)

# 提取关键词
df_keywords = df[["appln_id", "Result"]].copy()
df_keywords["keywords"] = df_keywords["Result"].apply(extract_keywords)
df_keywords = df_keywords[["appln_id", "keywords"]]

# 提取创新点
df_innovations = df[["appln_id", "Result"]].copy()
df_innovations["innovations"] = df_innovations["Result"].apply(extract_innovations)
df_innovations = df_innovations[["appln_id", "innovations"]]

# 保存结果
df_keywords.to_csv(keywords_output_path, index=False)
df_innovations.to_csv(innovations_output_path, index=False)

print(f"关键词提取完成  已保存到: {keywords_output_path}")
print(f"创新点提取完成  已保存到: {innovations_output_path}")