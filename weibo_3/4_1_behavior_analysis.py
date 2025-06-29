#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
行为识别脚本
按照论文方案实现四类行为识别：基础设施中断、基础设施修复、政府救助、公众行为
"""

import pandas as pd
import numpy as np
import jieba
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
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
    
    logger = logging.getLogger('behavior_analysis')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = f'data/log/4_1_behavior_analysis_{timestamp}.log'
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

class BehaviorAnalyzer:
    def __init__(self):
        """初始化行为分析器"""
        self.behavior_keywords = self._build_behavior_keywords()
        self.model = None
        self.vectorizer = None
        
    def _build_behavior_keywords(self):
        """构建改进的行为关键词词典"""
        logger.info("构建改进的行为关键词词典...")
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
        logger.info(f"改进的行为关键词词典构建完成，包含 {sum(len(cat) for behavior in behavior_keywords.values() for cat in behavior.values())} 个关键词")
        return behavior_keywords
    
    def segment_text(self, text):
        """文本分词"""
        if pd.isna(text):
            return []
        words = jieba.lcut(str(text))
        return [word for word in words if word.strip()]
    
    def extract_behavior_features(self, text):
        """提取行为特征"""
        words = self.segment_text(text)
        text_lower = str(text).lower()
        
        # 初始化特征向量
        features = {
            'emergency_help': 0,
            'info_search': 0,
            'mutual_support': 0
        }
        
        # 计算各类行为的特征得分
        for behavior_type, categories in self.behavior_keywords.items():
            score = 0
            for category, keywords in categories.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        score += 1
            # 简化类别名称
            if '紧急求助' in behavior_type:
                features['emergency_help'] = score
            elif '信息搜索' in behavior_type:
                features['info_search'] = score
            elif '互助支援' in behavior_type:
                features['mutual_support'] = score
        
        return features
    
    def create_training_data(self, df, sample_size=2000):
        """创建训练数据（基于改进的分类）"""
        logger.info("创建改进的训练数据...")
        
        # 随机抽样
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        # 基于改进的关键词规则进行标注
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
        logger.info(f"训练数据创建完成，样本数量: {len(sample_df)}")
        return sample_df
    
    def train_behavior_model(self, training_data):
        """训练行为识别模型"""
        logger.info("训练行为识别模型...")
        
        # 准备训练数据
        X = training_data['txt'].fillna('')
        y = training_data['behavior_label']
        
        # 使用TF-IDF特征提取
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        X_tfidf = self.vectorizer.fit_transform(X)
        
        # 训练随机森林模型
        self.model = RandomForestClassifier(
            n_estimators=100,
            criterion='gini',
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = self.model.predict(X_test)
        
        # 计算评估指标
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        evaluation_results = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        logger.info(f"行为识别模型评估结果:")
        logger.info(f"准确率: {precision:.3f}")
        logger.info(f"召回率: {recall:.3f}")
        logger.info(f"F1分数: {f1:.3f}")
        
        # 详细分类报告
        logger.info("\n详细分类报告:")
        logger.info(classification_report(y_test, y_pred))
        
        return evaluation_results, X_test, y_test, y_pred
    
    def predict_behaviors(self, df):
        """预测整个数据集的行为类型"""
        logger.info("预测行为类型...")
        
        if self.model is None or self.vectorizer is None:
            raise ValueError("模型未训练，请先调用train_behavior_model方法")
        
        # 特征提取
        X = df['txt'].fillna('')
        X_tfidf = self.vectorizer.transform(X)
        
        # 预测
        predictions = self.model.predict(X_tfidf)
        probabilities = self.model.predict_proba(X_tfidf)
        
        # 添加预测结果到数据框
        df_with_behavior = df.copy()
        df_with_behavior['predicted_behavior'] = predictions
        df_with_behavior['behavior_confidence'] = np.max(probabilities, axis=1)
        
        # 添加各类行为的概率
        behavior_classes = self.model.classes_
        for i, behavior in enumerate(behavior_classes):
            df_with_behavior[f'{behavior}_probability'] = probabilities[:, i]
        
        return df_with_behavior
    
    def analyze_behavior_distribution(self, df_with_behavior):
        """分析行为分布"""
        logger.info("分析行为分布...")
        
        # 创建结果目录
        os.makedirs('result/behavior', exist_ok=True)
        
        # 行为类型分布
        behavior_counts = df_with_behavior['predicted_behavior'].value_counts()
        
        plt.figure(figsize=(15, 10))
        
        # 子图1: 行为类型分布饼图
        plt.subplot(2, 3, 1)
        plt.pie(behavior_counts.values, labels=behavior_counts.index, autopct='%1.1f%%')
        plt.title('行为类型分布')
        
        # 子图2: 行为类型分布柱状图
        plt.subplot(2, 3, 2)
        behavior_counts.plot(kind='bar')
        plt.title('行为类型分布')
        plt.xticks(rotation=45)
        
        # 子图3: 用户类型与行为类型的关系
        plt.subplot(2, 3, 3)
        cross_tab = pd.crosstab(df_with_behavior['user_type'], df_with_behavior['predicted_behavior'])
        cross_tab.plot(kind='bar', stacked=True)
        plt.title('用户类型与行为类型关系')
        plt.xticks(rotation=45)
        plt.legend(title='行为类型', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 子图4: 行为置信度分布
        plt.subplot(2, 3, 4)
        plt.hist(df_with_behavior['behavior_confidence'], bins=20, alpha=0.7, color='lightgreen')
        plt.title('行为识别置信度分布')
        plt.xlabel('置信度')
        plt.ylabel('频次')
        
        # 子图5: 各类行为概率分布
        plt.subplot(2, 3, 5)
        behavior_prob_cols = [col for col in df_with_behavior.columns if col.endswith('_probability')]
        if behavior_prob_cols:
            prob_data = df_with_behavior[behavior_prob_cols]
            prob_data.boxplot()
            plt.title('各类行为概率分布')
            plt.xticks(rotation=45)
        
        # 子图6: 行为类型与情感类型的关系（如果有情感数据）
        plt.subplot(2, 3, 6)
        if 'sentiment_type' in df_with_behavior.columns:
            sentiment_behavior = pd.crosstab(df_with_behavior['sentiment_type'], df_with_behavior['predicted_behavior'])
            sentiment_behavior.plot(kind='bar', stacked=True)
            plt.title('情感类型与行为类型关系')
            plt.xticks(rotation=45)
            plt.legend(title='行为类型', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.text(0.5, 0.5, '无情感数据', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('情感类型与行为类型关系')
        
        plt.tight_layout()
        plt.savefig('result/behavior/behavior_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存行为分析结果
        df_with_behavior.to_csv('result/behavior/behavior_analysis_results.csv', 
                               index=False, encoding='utf-8')
        
        # 生成行为分析报告
        self.generate_behavior_report(df_with_behavior)
    
    def generate_behavior_report(self, df_with_behavior):
        """生成行为分析报告"""
        behavior_counts = df_with_behavior['predicted_behavior'].value_counts()
        
        report = f"""
行为分析报告
============

总体统计:
- 总文本数: {len(df_with_behavior)}
- 行为类型分布:
{'-' * 40}
"""
        
        for behavior, count in behavior_counts.items():
            percentage = (count / len(df_with_behavior)) * 100
            report += f"{behavior}: {count} ({percentage:.1f}%)\n"
        
        report += f"""
详细分析:
{'-' * 40}

1. 紧急求助行为:
- 数量: {len(df_with_behavior[df_with_behavior['predicted_behavior'] == '紧急求助行为'])}
- 平均置信度: {df_with_behavior[df_with_behavior['predicted_behavior'] == '紧急求助行为']['behavior_confidence'].mean():.3f}
- 子类别: 生命安全求助、生活需求求助、信息需求求助

2. 信息搜索行为:
- 数量: {len(df_with_behavior[df_with_behavior['predicted_behavior'] == '信息搜索行为'])}
- 平均置信度: {df_with_behavior[df_with_behavior['predicted_behavior'] == '信息搜索行为']['behavior_confidence'].mean():.3f}
- 子类别: 主动询问、信息求证、关注动态、寻求建议

3. 互助支援行为:
- 数量: {len(df_with_behavior[df_with_behavior['predicted_behavior'] == '互助支援行为'])}
- 平均置信度: {df_with_behavior[df_with_behavior['predicted_behavior'] == '互助支援行为']['behavior_confidence'].mean():.3f}
- 子类别: 紧急救援、物资支援、志愿服务、情感支持

用户类型行为分析:
{'-' * 40}
"""
        
        user_behavior = pd.crosstab(df_with_behavior['user_type'], df_with_behavior['predicted_behavior'])
        report += user_behavior.to_string()
        
        report += f"""

行为置信度统计:
{'-' * 40}
- 平均行为置信度: {df_with_behavior['behavior_confidence'].mean():.3f}
- 行为置信度标准差: {df_with_behavior['behavior_confidence'].std():.3f}
- 最高行为置信度: {df_with_behavior['behavior_confidence'].max():.3f}
- 最低行为置信度: {df_with_behavior['behavior_confidence'].min():.3f}
- 高置信度样本(>0.8): {len(df_with_behavior[df_with_behavior['behavior_confidence'] > 0.8])} ({len(df_with_behavior[df_with_behavior['behavior_confidence'] > 0.8]) / len(df_with_behavior) * 100:.1f}%)
"""
        
        with open('result/behavior/behavior_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info("行为分析报告已保存到 result/behavior/behavior_analysis_report.txt")

def main():
    """主函数"""
    logger.info("=== 4. 行为分析开始 ===")
    
    # 加载数据
    try:
        df = pd.read_csv('data/clean/processed_data.csv')
        logger.info("加载清洗后数据")
    except:
        logger.info("数据加载失败，请先运行数据预处理脚本")
        return
    
    logger.info(f"数据规模: {len(df)} 条记录")
    
    # 初始化行为分析器
    analyzer = BehaviorAnalyzer()
    
    # 创建训练数据
    training_data = analyzer.create_training_data(df, sample_size=2000)
    
    # 训练模型
    evaluation_results, X_test, y_test, y_pred = analyzer.train_behavior_model(training_data)
    
    # 预测整个数据集
    df_with_behavior = analyzer.predict_behaviors(df)
    
    # 分析行为分布
    analyzer.analyze_behavior_distribution(df_with_behavior)
    
    # 保存评估结果
    evaluation_df = pd.DataFrame([evaluation_results])
    evaluation_df.to_csv('result/behavior/behavior_evaluation.csv', index=False)
    
    logger.info("=== 4. 行为分析完成 ===")
    return df_with_behavior, evaluation_results

if __name__ == "__main__":
    main() 