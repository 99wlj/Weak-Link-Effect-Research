# 弱链接效应研究 - 专利关键词网络分析

## 📋 项目概述

基于专利关键词共现网络，研究弱链接向强链接演化的预测模型。通过构建时间窗口内的关键词网络，提取多维度特征，预测弱连接是否会在未来演化为强连接。

## 🏗️ 项目结构

```
📦Weak-Link-Effect-Research
├── preprocessing/                     # 数据预处理
│   └── Extract_refine_keywords_innovation.py
├── network/                          # 网络构建
│   ├── co_occurrence_links.py
│   └── keyword_co-occurrence_network_bywindows.py
├── feature_engineering/              # 特征工程
│   ├── construct_target_variables.py
│   ├── conduct_dependent_variableX1.py
│   ├── conduct_dependent_variableX2.py
│   └── conduct_dependent_variableX3.py
├── balance_dataset.py                # 数据平衡
└── data/                            # 数据目录
```

## 📊 数据流程

### 1. 数据预处理
**脚本**: `Extract_refine_keywords_innovation.py`
- **输入**: `my_raw_data.csv` (原始专利数据)
- **输出**: 
  - `Keywords_Processed_[timestamp].csv` - 提取的关键词
  - `Innovations_Extracted_[timestamp].csv` - 创新点
- **功能**: 词形还原、同义词映射、关键词标准化

### 2. 网络构建
**脚本**: `keyword_co-occurrence_network_bywindows.py`
- **输入**: `Keywords_Processed_*.csv`
- **输出**: `Keyword_LinkStrength_ByWindow_*.csv`
- **参数**: 
  - 时间窗口长度: 3年
  - 滑动步长: 2年
- **功能**: 构建时间窗口内的关键词共现网络，计算链接强度

### 3. 目标变量构建
**脚本**: `construct_target_variables.py`
- **输入**: `Keyword_LinkStrength_ByWindow_*.csv`
- **输出**: `target_variables_Y*.csv`
- **功能**: 判断弱连接是否在下一时间窗口演化为强连接 (Y=0/1)

### 4. 数据平衡
**脚本**: `balance_dataset.py`
- **输入**: `target_variables_Y*.csv`
- **输出**: `RES01_patent_based_sampled_*.csv`
- **策略**: 基于边涉及的专利数进行智能采样
- **目标**: 每个时间窗口3000个样本

### 5. 特征计算

#### X1 - 基本网络特征 (4个)
**脚本**: `conduct_dependent_variableX1.py`
- **输入**: `RES01_patent_based_sampled_*.csv`
- **输出**: `RES02_features_X1_optimized_*.csv`
- **特征**:
  - `link_strength`: 链接强度
  - `degree_difference`: 度差（节点度数差的绝对值）
  - `betweenness`: 边介数中心性
  - `tech_distance`: 技术距离（1-Jaccard相似度）

#### X2 - 扩展网络特征 (4个)
**脚本**: `conduct_dependent_variableX2.py`
- **输入**: `RES02_features_X1_optimized_*.csv`
- **输出**: `RES03_features_X1X2_complete_*.csv`
- **特征**:
  - `common_neighbors`: 共同邻居数
  - `adamic_adar`: Adamic-Adar指数
  - `resource_allocation`: 资源分配指数
  - `preferential_attachment`: 优先连接指数

#### X3 - 关联特征 (10个)
**脚本**: `conduct_dependent_variableX3.py`
- **输入**: `RES03_features_X1X2_complete_*.csv`
- **输出**: `RES04_features_X1X2X3_complete_*.csv`
- **特征**:
  - 创新点语义网络特征 (5个): 密度、聚类系数、直径、度中心性标准差、模块度
  - 发明人合作网络特征 (5个): 密度、聚类系数、直径、度中心性标准差、模块度

## 🔧 使用方法

### 环境要求
```python
pandas>=1.3.0
numpy>=1.20.0
networkx>=2.6
scikit-learn>=0.24.0
sentence-transformers>=2.0.0
python-louvain>=0.15
nltk>=3.6
tqdm>=4.60.0
matplotlib>=3.3.0
```

### 执行顺序
```bash
# 1. 数据预处理
python preprocessing/Extract_refine_keywords_innovation.py

# 2. 构建网络
python network/keyword_co-occurrence_network_bywindows.py

# 3. 构建目标变量
python feature_engineering/construct_target_variables.py

# 4. 数据平衡
python balance_dataset.py

# 5. 特征计算（按顺序执行）
python feature_engineering/conduct_dependent_variableX1.py
python feature_engineering/conduct_dependent_variableX2.py
python feature_engineering/conduct_dependent_variableX3.py
```

## 📈 指标说明

### 链接强度分类
- **强连接**: 链接强度 ≥ 中位数
- **弱连接**: 链接强度 < 中位数

### 目标变量
- **Y=1**: 弱连接在下一时间窗口演化为强连接
- **Y=0**: 弱连接保持为弱连接

### 特征标准化
所有特征都标准化到[0,1]区间，原始值保存在`*_raw`列中

## 📁 输出文件格式

### 最终特征文件
`RES04_features_X1X2X3_complete_*.csv`
- 包含18个标准化特征
- 样本标识信息（窗口、节点对）
- 目标变量Y
- 专利数等级

### 报告文件
每个阶段都生成对应的`*_report_*.txt`文件，包含：
- 数据统计
- 特征分布
- Y=0/1对比分析
- 特征相关性

## ⚙️ 配置参数

在各脚本开头可修改：
- `DATA_DIR`: 数据目录路径
- `window_length`: 时间窗口长度（默认3年）
- `step_size`: 滑动步长（默认2年）
- `TARGET_SAMPLES_PER_WINDOW`: 每窗口目标样本数（默认3000）

## 📝 注意事项

1. 确保`data/`目录下有所需的输入文件
2. X3特征计算需要下载语义模型（首次运行会自动下载）
3. 大规模网络的边介数计算可能耗时较长
4. 所有特征计算都使用了向量化操作优化性能