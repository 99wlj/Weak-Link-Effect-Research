import pandas as pd
import re
from datetime import datetime
import os
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk

# 确保下载必要的 NLTK 数据
nltk.download('wordnet')
nltk.download('omw-1.4')

# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据文件路径
file_path = os.path.join(BASE_DIR, "data", "my_raw_data.csv")

# 输出文件路径
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
keywords_output_path = os.path.join(BASE_DIR, "data", f"Keywords_Processed_{timestamp}.csv")
innovations_output_path = os.path.join(BASE_DIR, "data", f"Innovations_Extracted_{timestamp}.csv")

# 初始化词干和词形还原器
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# 无人机领域同义词映射
synonym_map = {
    "uav": "drone",
    "drones": "drone",
    "unmanned aerial vehicle": "drone",
    "uas": "drone",
    "flight controller": "fc",
    "autopilot": "fc",
    "gps": "gps navigation",
    "gnss": "gps navigation",
    "imu": "inertial measurement unit",
    "lidar": "lidar sensor",
    "laser scanner": "lidar sensor",
    "camera": "optical camera",
    "vision sensor": "optical camera",
    "thermal camera": "infrared camera",
    "wifi": "wireless communication",
    "4g": "cellular communication",
    "5g": "cellular communication",
    "radio": "radio communication",
    "rf": "radio communication",
    "ai": "artificial intelligence",
    "ml": "machine learning",
    "deep learning": "deep learning",
    "dl": "deep learning",
    "cnn": "convolutional neural network",
    "rnn": "recurrent neural network",
    "pid": "pid control",
    "quadcopter": "multirotor",
    "hexacopter": "multirotor",
    "octocopter": "multirotor",
    "fixed wing": "fixed-wing drone",
    "surveillance": "monitoring",
    "inspection": "monitoring",
    "mapping": "surveying",
    "photogrammetry": "surveying",
    "fc": "flight controller",
    "gps nav": "gps navigation",
    "imu sensor": "inertial measurement unit"
}

def normalize_keyword(keyword):
    """
    1. 转小写
    2. 拆分多词短语
    3. 词干还原 + 词形还原
    4. 拼回短语
    5. 同义词消歧义
    """
    kw = keyword.lower().strip()
    words = kw.split()
    # 先词形还原再词干
    words = [stemmer.stem(lemmatizer.lemmatize(w)) for w in words]
    normalized = ' '.join(words)
    # 同义词消歧义
    normalized = synonym_map.get(normalized, normalized)
    return normalized

# 改进的关键词提取函数
def extract_keywords(result_str):
    if pd.isna(result_str):
        return []
    keyword_match = re.search(r'Keywords:\s*\[(.*?)\]', result_str)
    if keyword_match:
        keywords_str = keyword_match.group(1)
        keywords = [normalize_keyword(k.strip()) for k in keywords_str.split(',')]
        return keywords
    return []

# 创新点提取函数（保持原样）
def extract_innovations(result_str):
    if pd.isna(result_str):
        return []
    innovation_match = re.search(r'Innovation:(.*?)(?=Keywords:|$)', result_str, re.DOTALL)
    if innovation_match:
        innovation_text = innovation_match.group(1).strip()
        innovations = re.findall(r'\d+\.\s*(.*?)(?=\d+\.|$)', innovation_text, re.DOTALL)
        if innovations:
            return [innovation.strip() for innovation in innovations if innovation.strip()]
        lines = innovation_text.split('\n')
        return [line.strip() for line in lines if line.strip() and not line.strip().startswith('Keywords:')]
    return []

# 读取 CSV
df = pd.read_csv(file_path)

# ========= 提取关键词 =========
df_keywords = df[["appln_id", "earliest_publn_year", "Result"]].copy()
df_keywords["keywords"] = df_keywords["Result"].apply(extract_keywords)
df_keywords = df_keywords[["appln_id", "earliest_publn_year", "keywords"]]
df_keywords.to_csv(keywords_output_path, index=False)

# ========= 提取创新点 =========
df_innovations = df[["appln_id", "earliest_publn_year", "Result"]].copy()
df_innovations["innovations"] = df_innovations["Result"].apply(extract_innovations)
df_innovations = df_innovations[["appln_id", "earliest_publn_year", "innovations"]]
df_innovations.to_csv(innovations_output_path, index=False)

print(f"关键词处理完成  已保存到: {keywords_output_path}")
print(f"创新点提取完成  已保存到: {innovations_output_path}")
