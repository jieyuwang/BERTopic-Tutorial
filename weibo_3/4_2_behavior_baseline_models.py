#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
行为识别基准模型对比脚本
使用多种机器学习模型进行行为分类，并对比性能
"""

import pandas as pd
import numpy as np
import jieba
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
import logging
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
def setup_logger():
    """设置日志记录器"""
    os.makedirs('data/log', exist_ok=True)
    
    logger = logging.getLogger('baseline_models')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = f'data/log/4_2_behavior_baseline_models_{timestamp}.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()

class BaselineModelAnalyzer:
    def __init__(self):
        """初始化基准模型分析器"""
        self.behavior_keywords = self._build_behavior_keywords()
        self.models = {}
        self.vectorizer = None
        
    def _build_behavior_keywords(self):
        """构建行为关键词词典"""
        behavior_keywords = {
            '紧急求助行为': {
                '生命安全求助': ['救命', '被困', '危险', '紧急', '快', '急'],
                '生活需求求助': ['求助', '需要帮助', '帮忙', '支持', '援助'],
                '信息需求求助': ['请问', '谁知道', '求问', '咨询', '了解']
            },
            '信息搜索行为': {
                '主动询问': ['请问', '谁知道', '求问', '帮忙', '需要', '了解'],
                '信息求证': ['真的吗', '确认', '核实', '求证', '是否', '真假'],
                '关注动态': ['关注', '跟踪', '更新', '最新', '情况', '进展'],
                '寻求建议': ['建议', '怎么办', '如何', '应该', '推荐', '意见']
            },
            '互助支援行为': {
                '紧急救援': ['救援', '抢险', '救助', '转移', '疏散'],
                '物资支援': ['捐赠', '支援', '提供', '分享', '免费'],
                '志愿服务': ['志愿者', '互助', '团结', '一起', '参与'],
                '情感支持': ['加油', '支持', '鼓励', '安慰', '理解']
            }
        }
        return behavior_keywords
    
    def create_training_data(self, df, sample_size=2000):
        """创建训练数据"""
        logger.info("创建训练数据...")
        
        # 随机抽样
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        # 基于关键词规则进行标注
        def label_behavior(text):
            text_lower = str(text).lower()
            
            # 计算各类行为的匹配度
            scores = {
                '紧急求助行为': 0,
                '信息搜索行为': 0,
                '互助支援行为': 0
            }
            
            for behavior_type, categories in self.behavior_keywords.items():
                for category, keywords in categories.items():
                    for keyword in keywords:
                        if keyword in text_lower:
                            scores[behavior_type] += 1
            
            # 返回得分最高的行为类型
            if max(scores.values()) == 0:
                return '信息搜索行为'  # 默认分类
            else:
                return max(scores.items(), key=lambda x: x[1])[0]
        
        sample_df['behavior_label'] = sample_df['txt'].apply(label_behavior)
        
        return sample_df
    
    def initialize_models(self):
        """初始化所有基准模型"""
        self.models = {
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'NaiveBayes': MultinomialNB(),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        logger.info(f"初始化了 {len(self.models)} 个基准模型")
    
    def train_and_evaluate_models(self, training_data):
        """训练和评估所有模型"""
        logger.info("开始训练和评估基准模型...")
        
        # 准备数据
        X = training_data['txt'].fillna('')
        y = training_data['behavior_label']
        
        # 特征提取
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        X_tfidf = self.vectorizer.fit_transform(X)
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 训练和评估每个模型
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"训练 {name} 模型...")
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            
            # 计算评估指标
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            # 交叉验证
            cv_scores = cross_val_score(model, X_tfidf, y, cv=5, scoring='f1_weighted')
            
            results[name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_test': y_test
            }
            
            logger.info(f"{name} - 准确率: {precision:.3f}, 召回率: {recall:.3f}, F1: {f1:.3f}")
        
        return results, X_test, y_test
    
    def create_model_comparison_visualization(self, results):
        """创建模型对比可视化"""
        logger.info("创建模型对比可视化...")
        
        # 创建结果目录
        os.makedirs('result/baseline_models', exist_ok=True)
        
        # 准备数据
        model_names = list(results.keys())
        metrics = ['precision', 'recall', 'f1_score', 'cv_mean']
        
        # 为每个指标创建单独的图片
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            
            values = [results[name][metric] for name in model_names]
            
            bars = plt.bar(model_names, values, alpha=0.7, color='skyblue')
            plt.title(f'{metric.replace("_", " ").title()} 对比', fontsize=14)
            plt.ylabel('Score', fontsize=12)
            plt.xlabel('Model', fontsize=12)
            plt.xticks(rotation=45)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(f'result/baseline_models/{metric}_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"{metric}对比图已保存到 result/baseline_models/{metric}_comparison.png")
        
        # 创建综合性能雷达图
        self.create_radar_chart(results)
        
        logger.info("所有模型对比可视化已完成")
    
    def create_radar_chart(self, results):
        """创建模型性能雷达图"""
        logger.info("创建模型性能雷达图...")
        
        # 准备数据
        model_names = list(results.keys())
        metrics = ['precision', 'recall', 'f1_score', 'cv_mean']
        
        # 设置雷达图的角度
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        # 创建雷达图
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))
        
        # 为每个模型绘制雷达图
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        for i, (model_name, result) in enumerate(results.items()):
            values = [result[metric] for metric in metrics]
            values += values[:1]  # 闭合图形
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([metric.replace('_', ' ').title() for metric in metrics])
        ax.set_ylim(0, 1)
        ax.set_title('模型性能雷达图对比', size=16, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig('result/baseline_models/models_radar_chart.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("雷达图已保存到 result/baseline_models/models_radar_chart.png")
    
    def generate_detailed_reports(self, results):
        """生成详细的模型报告"""
        logger.info("生成详细模型报告...")
        
        # 创建结果表格
        report_data = []
        for name, result in results.items():
            report_data.append({
                'Model': name,
                'Precision': f"{result['precision']:.3f}",
                'Recall': f"{result['recall']:.3f}",
                'F1-Score': f"{result['f1_score']:.3f}",
                'CV Mean': f"{result['cv_mean']:.3f}",
                'CV Std': f"{result['cv_std']:.3f}"
            })
        
        results_df = pd.DataFrame(report_data)
        results_df.to_csv('result/baseline_models/baseline_models_results.csv', index=False)
        
        # 生成详细报告
        report = f"""
基准模型对比分析报告
====================

模型性能对比:
{'-' * 50}
{results_df.to_string(index=False)}

详细分析:
{'-' * 50}
"""
        
        # 找出最佳模型
        best_f1_model = max(results.items(), key=lambda x: x[1]['f1_score'])
        best_cv_model = max(results.items(), key=lambda x: x[1]['cv_mean'])
        
        report += f"""
最佳模型分析:
{'-' * 50}
- 最佳F1分数模型: {best_f1_model[0]} (F1: {best_f1_model[1]['f1_score']:.3f})
- 最佳交叉验证模型: {best_cv_model[0]} (CV Mean: {best_cv_model[1]['cv_mean']:.3f})

模型选择建议:
{'-' * 50}
- 如果关注F1分数: 推荐使用 {best_f1_model[0]}
- 如果关注稳定性: 推荐使用 {best_cv_model[0]}
- 如果关注训练速度: 推荐使用 NaiveBayes
- 如果关注可解释性: 推荐使用 DecisionTree
"""
        
        with open('result/baseline_models/baseline_models_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info("详细模型报告已保存到 result/baseline_models/baseline_models_report.txt")
        
        return best_f1_model[0], best_cv_model[0]
    
    def predict_with_best_model(self, df, best_model_name):
        """使用最佳模型进行预测"""
        print(f"使用最佳模型 {best_model_name} 进行预测...")
        
        if best_model_name not in self.models:
            raise ValueError(f"模型 {best_model_name} 不存在")
        
        # 特征提取
        X = df['txt'].fillna('')
        X_tfidf = self.vectorizer.transform(X)
        
        # 预测
        model = self.models[best_model_name]
        predictions = model.predict(X_tfidf)
        probabilities = model.predict_proba(X_tfidf)
        
        # 添加预测结果
        df_with_behavior = df.copy()
        df_with_behavior['predicted_behavior'] = predictions
        df_with_behavior['behavior_confidence'] = np.max(probabilities, axis=1)
        
        # 添加各类行为的概率
        behavior_classes = model.classes_
        for i, behavior in enumerate(behavior_classes):
            df_with_behavior[f'{behavior}_probability'] = probabilities[:, i]
        
        return df_with_behavior

def main():
    """主函数"""
    logger.info("=== 4_2. 基准模型对比分析开始 ===")
    
    # 加载数据
    try:
        df = pd.read_csv('data/clean/processed_data.csv')
        logger.info("加载清洗后数据")
    except:
        logger.info("数据加载失败，请先运行数据预处理脚本")
        return
    
    logger.info(f"数据规模: {len(df)} 条记录")
    
    # 初始化基准模型分析器
    analyzer = BaselineModelAnalyzer()
    
    # 创建训练数据
    training_data = analyzer.create_training_data(df, sample_size=2000)
    
    # 初始化模型
    analyzer.initialize_models()
    
    # 训练和评估所有模型
    results, X_test, y_test = analyzer.train_and_evaluate_models(training_data)
    
    # 创建可视化对比
    analyzer.create_model_comparison_visualization(results)
    
    # 生成详细报告
    best_f1_model, best_cv_model = analyzer.generate_detailed_reports(results)
    
    # 使用最佳模型进行预测
    df_with_behavior = analyzer.predict_with_best_model(df, best_f1_model)
    
    # 保存最佳模型结果
    df_with_behavior.to_csv(f'result/baseline_models/{best_f1_model.lower()}_behavior_results.csv', 
                            index=False, encoding='utf-8')
    
    logger.info(f"=== 4_2. 基准模型对比分析完成 ===")
    logger.info(f"最佳模型: {best_f1_model}")
    logger.info(f"结果已保存到: result/baseline_models/{best_f1_model.lower()}_behavior_results.csv")
    
    return results, df_with_behavior

if __name__ == "__main__":
    main() 