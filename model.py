"""
弱链接演化预测模型训练与评估
对比多个机器学习模型，评估在不同时间窗口的预测效果
"""

import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
import pickle
import json
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
warnings.filterwarnings('ignore')

# 配置
DATA_DIR = 'F:/WLJ/Weak-Link-Effect-Research/data'
MODEL_DIR = os.path.join(DATA_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# 设置随机种子
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 80)
print("弱链接演化预测模型训练与评估")
print("=" * 80)

# 1. 加载数据
print("\n[1] 加载特征数据...")

# 查找最新的X1X2X3特征文件
feature_files = [f for f in os.listdir(DATA_DIR) if f.startswith('RES04_features_X1X2X3_complete_')]
if not feature_files:
    raise FileNotFoundError("未找到X1X2X3特征文件！")

latest_feature_file = sorted(feature_files)[-1]
feature_path = os.path.join(DATA_DIR, latest_feature_file)
print(f"  使用特征文件: {latest_feature_file}")

df = pd.read_csv(feature_path)
print(f"  样本总数: {len(df):,}")
print(f"  正样本(Y=1): {(df['y']==1).sum():,} ({(df['y']==1).sum()/len(df)*100:.1f}%)")
print(f"  负样本(Y=0): {(df['y']==0).sum():,} ({(df['y']==0).sum()/len(df)*100:.1f}%)")

# 2. 特征选择
print("\n[2] 准备特征...")

# 定义特征列
x1_features = ['link_strength', 'degree_difference', 'betweenness', 'tech_distance']
x2_features = ['common_neighbors', 'adamic_adar', 'resource_allocation', 'preferential_attachment']
x3_features = [
    'innovation_density', 'innovation_avg_clustering', 'innovation_diameter',
    'innovation_degree_centrality_std', 'innovation_modularity',
    'inventor_density', 'inventor_avg_clustering', 'inventor_diameter', 
    'inventor_degree_centrality_std', 'inventor_modularity'
]

all_features = x1_features + x2_features + x3_features
print(f"  特征总数: {len(all_features)}")
print(f"    - X1特征: {len(x1_features)}个")
print(f"    - X2特征: {len(x2_features)}个")
print(f"    - X3特征: {len(x3_features)}个")

# 检查特征完整性
missing_features = [f for f in all_features if f not in df.columns]
if missing_features:
    print(f"  ⚠️ 缺失特征: {missing_features}")
    all_features = [f for f in all_features if f in df.columns]

# 准备特征矩阵和目标变量
X = df[all_features].values
y = df['y'].values

# 获取时间窗口信息
df['window_id'] = df['window_t_start'].astype(str) + '-' + df['window_t_end'].astype(str)
unique_windows = df['window_id'].unique()
print(f"\n  时间窗口数: {len(unique_windows)}")
for window in sorted(unique_windows):
    window_samples = df[df['window_id'] == window]
    print(f"    {window}: {len(window_samples):,} 样本 "
          f"(Y=1: {(window_samples['y']==1).sum():,})")

# 3. 定义模型
print("\n[3] 初始化模型...")

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, 
        random_state=RANDOM_STATE,
        class_weight='balanced'
    ),
    
    'Decision Tree': DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
        class_weight='balanced'
    ),
    
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        n_jobs=-1
    ),
    
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=RANDOM_STATE
    ),
    
    'XGBoost': xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=(y==0).sum()/(y==1).sum()  # 处理不平衡
    ),
    
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        verbosity=-1
    )
}

print(f"  准备训练 {len(models)} 个模型")

# 4. 整体数据集划分
print("\n[4] 数据集划分...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"  训练集: {len(X_train):,} 样本")
print(f"  测试集: {len(X_test):,} 样本")

# 5. 模型训练与评估
print("\n[5] 训练模型...")

def evaluate_model(y_true, y_pred, y_prob=None):
    """计算所有评估指标"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc_roc': roc_auc_score(y_true, y_prob) if y_prob is not None else 0
    }
    return metrics

# 存储结果
overall_results = {}
window_results = {}
best_model = None
best_score = 0

# 训练每个模型
for model_name, model in tqdm(models.items(), desc="训练模型"):
    print(f"\n  训练 {model_name}...")
    
    try:
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 获取概率预测（如果可用）
        if hasattr(model, 'predict_proba'):
            y_prob_train = model.predict_proba(X_train)[:, 1]
            y_prob_test = model.predict_proba(X_test)[:, 1]
        else:
            y_prob_train = y_prob_test = None
        
        # 评估整体性能
        train_metrics = evaluate_model(y_train, y_pred_train, y_prob_train)
        test_metrics = evaluate_model(y_test, y_pred_test, y_prob_test)
        
        # 5折交叉验证
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        
        overall_results[model_name] = {
            'train': train_metrics,
            'test': test_metrics,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        # 检查是否为最佳模型
        if test_metrics['f1'] > best_score:
            best_score = test_metrics['f1']
            best_model = (model_name, model)
        
        print(f"    测试集 - F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['auc_roc']:.4f}")
        
    except Exception as e:
        print(f"    ❌ 训练失败: {e}")
        overall_results[model_name] = None

# 6. 时间窗口评估
print("\n[6] 时间窗口评估...")

# 对最佳模型进行时间窗口评估
if best_model:
    model_name, model = best_model
    print(f"  使用最佳模型: {model_name}")
    
    for window in sorted(unique_windows):
        # 获取该窗口的数据
        window_mask = df['window_id'] == window
        X_window = df[window_mask][all_features].values
        y_window = df[window_mask]['y'].values
        
        if len(y_window) < 50:  # 样本太少，跳过
            continue
        
        # 划分训练测试集
        if len(np.unique(y_window)) > 1:  # 确保有两个类别
            X_w_train, X_w_test, y_w_train, y_w_test = train_test_split(
                X_window, y_window, test_size=0.3, random_state=RANDOM_STATE, stratify=y_window
            )
            
            # 训练
            temp_model = models[model_name].__class__(**models[model_name].get_params())
            temp_model.fit(X_w_train, y_w_train)
            
            # 预测
            y_w_pred = temp_model.predict(X_w_test)
            y_w_prob = temp_model.predict_proba(X_w_test)[:, 1] if hasattr(temp_model, 'predict_proba') else None
            
            # 评估
            window_metrics = evaluate_model(y_w_test, y_w_pred, y_w_prob)
            
            if model_name not in window_results:
                window_results[model_name] = {}
            window_results[model_name][window] = window_metrics

# 7. 生成评估报告
print("\n[7] 生成评估报告...")

# 创建结果DataFrame
results_data = []
for model_name, results in overall_results.items():
    if results:
        row = {'Model': model_name}
        row.update({f'Train_{k}': v for k, v in results['train'].items()})
        row.update({f'Test_{k}': v for k, v in results['test'].items()})
        row['CV_Mean_F1'] = results['cv_mean']
        row['CV_Std_F1'] = results['cv_std']
        results_data.append(row)

df_results = pd.DataFrame(results_data)
df_results = df_results.sort_values('Test_f1', ascending=False)

print("\n" + "=" * 80)
print("模型评估结果（整体）")
print("=" * 80)
print(df_results.to_string(index=False))

# 时间窗口结果
if window_results:
    print("\n" + "=" * 80)
    print(f"时间窗口评估结果 - {best_model[0]}")
    print("=" * 80)
    
    window_df_data = []
    for window in sorted(unique_windows):
        if best_model[0] in window_results and window in window_results[best_model[0]]:
            metrics = window_results[best_model[0]][window]
            row = {'Window': window}
            row.update(metrics)
            window_df_data.append(row)
    
    if window_df_data:
        df_window_results = pd.DataFrame(window_df_data)
        print(df_window_results.to_string(index=False))
        
        # 计算平均性能
        print("\n时间窗口平均性能:")
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']:
            if metric in df_window_results.columns:
                mean_val = df_window_results[metric].mean()
                std_val = df_window_results[metric].std()
                print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")

# 8. 保存最佳模型
if best_model:
    model_name, model = best_model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存模型
    model_filename = f"best_model_{model_name.replace(' ', '_')}_{timestamp}.pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n✅ 最佳模型已保存: {model_filename}")
    print(f"   模型: {model_name}")
    print(f"   测试集F1分数: {overall_results[model_name]['test']['f1']:.4f}")
    
    # 保存模型配置和性能
    config = {
        'model_name': model_name,
        'features': all_features,
        'feature_count': len(all_features),
        'timestamp': timestamp,
        'performance': overall_results[model_name],
        'data_info': {
            'total_samples': len(df),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'positive_ratio': (y==1).sum() / len(y)
        }
    }
    
    config_filename = f"model_config_{timestamp}.json"
    config_path = os.path.join(MODEL_DIR, config_filename)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"   配置文件: {config_filename}")

# 9. 特征重要性分析（针对树模型）
if best_model and hasattr(best_model[1], 'feature_importances_'):
    print("\n[8] 特征重要性分析...")
    
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'importance': best_model[1].feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 重要特征:")
    print(feature_importance.head(10).to_string(index=False))
    
    # 按特征组统计
    x1_importance = feature_importance[feature_importance['feature'].isin(x1_features)]['importance'].sum()
    x2_importance = feature_importance[feature_importance['feature'].isin(x2_features)]['importance'].sum()
    x3_importance = feature_importance[feature_importance['feature'].isin(x3_features)]['importance'].sum()
    
    print(f"\n特征组重要性:")
    print(f"  X1特征总重要性: {x1_importance:.4f}")
    print(f"  X2特征总重要性: {x2_importance:.4f}")
    print(f"  X3特征总重要性: {x3_importance:.4f}")
    
    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance.head(15)['feature'], feature_importance.head(15)['importance'])
    plt.xlabel('Importance')
    plt.title(f'Top 15 Feature Importance - {best_model[0]}')
    plt.tight_layout()
    
    # 保存图片
    plot_filename = f"feature_importance_{timestamp}.png"
    plt.savefig(os.path.join(MODEL_DIR, plot_filename), dpi=100, bbox_inches='tight')
    print(f"  特征重要性图保存为: {plot_filename}")
    plt.close()

# 10. 生成详细评估报告
print("\n[9] 生成详细评估报告...")

report_filename = f"model_evaluation_report_{timestamp}.txt"
report_path = os.path.join(MODEL_DIR, report_filename)

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("弱链接演化预测模型评估报告\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n\n")
    
    # 数据概况
    f.write("【数据概况】\n")
    f.write(f"数据文件: {latest_feature_file}\n")
    f.write(f"样本总数: {len(df):,}\n")
    f.write(f"正样本数: {(df['y']==1).sum():,} ({(df['y']==1).sum()/len(df)*100:.1f}%)\n")
    f.write(f"负样本数: {(df['y']==0).sum():,} ({(df['y']==0).sum()/len(df)*100:.1f}%)\n")
    f.write(f"特征数量: {len(all_features)}\n")
    f.write(f"时间窗口数: {len(unique_windows)}\n\n")
    
    # 特征列表
    f.write("【特征列表】\n")
    f.write(f"X1特征 ({len(x1_features)}个): {', '.join(x1_features)}\n")
    f.write(f"X2特征 ({len(x2_features)}个): {', '.join(x2_features)}\n")
    f.write(f"X3特征 ({len(x3_features)}个): {', '.join(x3_features)}\n\n")
    
    # 模型性能对比
    f.write("【模型性能对比】\n")
    f.write(df_results.to_string(index=False))
    f.write("\n\n")
    
    # 最佳模型详情
    if best_model:
        f.write("【最佳模型】\n")
        f.write(f"模型名称: {best_model[0]}\n")
        f.write(f"测试集性能:\n")
        for metric, value in overall_results[best_model[0]]['test'].items():
            f.write(f"  {metric}: {value:.4f}\n")
        f.write(f"交叉验证F1: {overall_results[best_model[0]]['cv_mean']:.4f} ± {overall_results[best_model[0]]['cv_std']:.4f}\n\n")
        
        # 时间窗口性能
        if window_results and best_model[0] in window_results:
            f.write("【时间窗口性能】\n")
            for window in sorted(unique_windows):
                if window in window_results[best_model[0]]:
                    metrics = window_results[best_model[0]][window]
                    f.write(f"\n窗口 {window}:\n")
                    for metric, value in metrics.items():
                        f.write(f"  {metric}: {value:.4f}\n")
        
        # 特征重要性
        if hasattr(best_model[1], 'feature_importances_'):
            f.write("\n【特征重要性Top10】\n")
            for _, row in feature_importance.head(10).iterrows():
                f.write(f"  {row['feature']}: {row['importance']:.4f}\n")

print(f"  详细报告已保存: {report_filename}")

# 11. 混淆矩阵和分类报告（针对最佳模型）
if best_model:
    print("\n[10] 生成混淆矩阵和分类报告...")
    
    # 重新预测测试集
    y_pred = best_model[1].predict(X_test)
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Weak→Weak', 'Weak→Strong'],
                yticklabels=['Weak→Weak', 'Weak→Strong'])
    plt.title(f'Confusion Matrix - {best_model[0]}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 添加准确率信息
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    plt.text(0.5, -0.1, f'Accuracy: {accuracy:.3f}', 
             ha='center', transform=plt.gca().transAxes)
    
    # 保存混淆矩阵图
    cm_filename = f"confusion_matrix_{timestamp}.png"
    plt.savefig(os.path.join(MODEL_DIR, cm_filename), dpi=100, bbox_inches='tight')
    print(f"  混淆矩阵已保存: {cm_filename}")
    plt.close()
    
    # 分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, 
                               target_names=['Weak→Weak', 'Weak→Strong'],
                               digits=4))
    
    # 12. ROC曲线
    if hasattr(best_model[1], 'predict_proba'):
        y_prob = best_model[1].predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc_score(y_test, y_prob):.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {best_model[0]}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # 保存ROC曲线
        roc_filename = f"roc_curve_{timestamp}.png"
        plt.savefig(os.path.join(MODEL_DIR, roc_filename), dpi=100, bbox_inches='tight')
        print(f"  ROC曲线已保存: {roc_filename}")
        plt.close()

# 13. 生成模型性能对比图
print("\n[11] 生成模型性能对比图...")

# 准备数据
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
model_names = []
metric_values = {metric: [] for metric in metrics_to_plot}

for model_name, results in overall_results.items():
    if results:
        model_names.append(model_name)
        for metric in metrics_to_plot:
            metric_values[metric].append(results['test'][metric])

# 创建对比图
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx]
    bars = ax.bar(range(len(model_names)), metric_values[metric])
    
    # 为最高值的柱子着色
    max_idx = np.argmax(metric_values[metric])
    bars[max_idx].set_color('green')
    
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'{metric.upper()} Comparison')
    ax.set_ylim([0, 1.05])
    
    # 添加数值标签
    for i, v in enumerate(metric_values[metric]):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    ax.grid(True, alpha=0.3, axis='y')

# 隐藏多余的子图
if len(metrics_to_plot) < 6:
    axes[-1].set_visible(False)

plt.suptitle('Model Performance Comparison', fontsize=16, y=1.02)
plt.tight_layout()

# 保存对比图
comparison_filename = f"model_comparison_{timestamp}.png"
plt.savefig(os.path.join(MODEL_DIR, comparison_filename), dpi=100, bbox_inches='tight')
print(f"  模型对比图已保存: {comparison_filename}")
plt.close()

# 14. 保存所有模型的预测结果（可选）
print("\n[12] 保存预测结果...")

predictions_df = pd.DataFrame({
    'y_true': y_test
})

for model_name, model in models.items():
    if model_name in overall_results and overall_results[model_name]:
        try:
            # 使用已训练的模型预测
            model.fit(X_train, y_train)  # 确保模型已训练
            y_pred = model.predict(X_test)
            predictions_df[f'{model_name}_pred'] = y_pred
            
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
                predictions_df[f'{model_name}_prob'] = y_prob
        except:
            pass

# 保存预测结果
predictions_filename = f"predictions_{timestamp}.csv"
predictions_df.to_csv(os.path.join(MODEL_DIR, predictions_filename), index=False)
print(f"  预测结果已保存: {predictions_filename}")

# 15. 生成最终摘要
print("\n" + "=" * 80)
print("模型训练完成 - 摘要")
print("=" * 80)

if best_model:
    print(f"\n🏆 最佳模型: {best_model[0]}")
    print(f"   - 测试集F1分数: {overall_results[best_model[0]]['test']['f1']:.4f}")
    print(f"   - 测试集AUC-ROC: {overall_results[best_model[0]]['test']['auc_roc']:.4f}")
    print(f"   - 测试集准确率: {overall_results[best_model[0]]['test']['accuracy']:.4f}")
    
    # 计算提升比例（相对于随机猜测）
    baseline_f1 = 2 * (y_test.sum()/len(y_test)) * 0.5 / ((y_test.sum()/len(y_test)) + 0.5)
    improvement = (overall_results[best_model[0]]['test']['f1'] - baseline_f1) / baseline_f1 * 100
    print(f"   - 相对基线提升: {improvement:.1f}%")

print(f"\n📁 输出文件位置: {MODEL_DIR}")
print(f"   - 模型文件: {model_filename}")
print(f"   - 配置文件: {config_filename}")
print(f"   - 评估报告: {report_filename}")
print(f"   - 混淆矩阵: {cm_filename}")
if 'roc_filename' in locals():
    print(f"   - ROC曲线: {roc_filename}")
print(f"   - 模型对比图: {comparison_filename}")
print(f"   - 预测结果: {predictions_filename}")

print("\n✅ 所有任务完成！")
print("=" * 80)

# 创建一个简单的模型使用示例
print("\n【模型使用示例】")
print("```python")
print("import pickle")
print("import numpy as np")
print("")
print(f"# 加载模型")
print(f"with open('{model_path}', 'rb') as f:")
print("    model = pickle.load(f)")
print("")
print("# 准备特征（18维）")
print(f"features = np.array([[...]])  # {len(all_features)}个特征")
print("")
print("# 预测")
print("prediction = model.predict(features)")
print("probability = model.predict_proba(features)[:, 1]")
print("")
print("print(f'预测结果: {prediction[0]}')")
print("print(f'演化为强链接概率: {probability[0]:.3f}')")
print("```")
