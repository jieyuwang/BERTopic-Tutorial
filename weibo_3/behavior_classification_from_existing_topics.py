#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于现有BERTopic主题建模结果的行为分类脚本
利用已训练好的主题模型进行行为分类
"""

import pandas as pd
import numpy as np
import jieba
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
import logging
import os
from datetime import datetime
import pickle
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
def setup_logger():
    """设置日志记录器"""
    os.makedirs('../weibo/data/log', exist_ok=True)
    
    logger = logging.getLogger('existing_topics_behavior_classification')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = f'data/log/existing_topics_behavior_classification_{timestamp}.log'
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

class ExistingTopicsBehaviorClassifier:
    def __init__(self):
        """初始化基于现有主题的行为分类器"""
        self.behavior_keywords = self._build_behavior_keywords()
        self.topic_behavior_mapping = {}
        self.classifier = None
        self.vectorizer = None
        
    def _build_behavior_keywords(self):
        """构建行为关键词词典"""
        logger.info("构建行为关键词词典...")
        behavior_keywords = {
            '紧急求助行为': {
                '生命安全求助': ['救命', '被困', '危险', '紧急', '快', '急', '救援', '转移', '疏散'],
                '生活需求求助': ['求助', '需要帮助', '帮忙', '支持', '援助', '食物', '水', '药品'],
                '信息需求求助': ['请问', '谁知道', '求问', '咨询', '了解', '情况', '最新']
            },
            '信息搜索行为': {
                '主动询问': ['请问', '谁知道', '求问', '帮忙', '需要', '了解', '什么', '哪里'],
                '信息求证': ['真的吗', '确认', '核实', '求证', '是否', '真假', '消息'],
                '关注动态': ['关注', '跟踪', '更新', '最新', '情况', '进展', '新闻'],
                '寻求建议': ['建议', '怎么办', '如何', '应该', '推荐', '意见', '方法']
            },
            '互助支援行为': {
                '紧急救援': ['救援', '抢险', '救助', '转移', '疏散', '志愿者', '参与'],
                '物资支援': ['捐赠', '支援', '提供', '分享', '免费', '物资', '帮助'],
                '志愿服务': ['志愿者', '互助', '团结', '一起', '参与', '服务'],
                '情感支持': ['加油', '支持', '鼓励', '安慰', '理解', '坚强']
            }
        }
        logger.info(f"行为关键词词典构建完成，包含 {sum(len(cat) for behavior in behavior_keywords.values() for cat in behavior.values())} 个关键词")
        return behavior_keywords
    
    def load_existing_results(self, topic_results_path, topic_info_path):
        """加载现有的主题建模结果"""
        logger.info("加载现有主题建模结果...")
        
        try:
            # 加载主题建模结果
            topic_results = pd.read_csv(topic_results_path)
            topic_info = pd.read_csv(topic_info_path)
            
            logger.info(f"主题建模结果加载成功")
            logger.info(f"- 文档数量: {len(topic_results)}")
            logger.info(f"- 主题数量: {len(topic_info)}")
            
            return topic_results, topic_info
            
        except Exception as e:
            logger.error(f"加载主题建模结果失败: {e}")
            return None, None
    
    def analyze_topic_keywords_for_behavior(self, topic_info, texts):
        """分析主题关键词，映射到行为类别"""
        logger.info("分析主题关键词，映射到行为类别...")
        
        topic_behavior_mapping = {}
        
        for _, topic_row in topic_info.iterrows():
            topic_id = topic_row['Topic']
            if topic_id == -1:  # 跳过异常值主题
                continue
            
            # 获取主题关键词（假设关键词在topic_info中）
            if 'Keywords' in topic_row:
                keywords = topic_row['Keywords'].split(', ')
            else:
                # 如果没有关键词列，使用主题ID作为标识
                keywords = [f"topic_{topic_id}"]
            
            # 计算该主题的行为得分
            behavior_scores = {
                '紧急求助行为': 0,
                '信息搜索行为': 0,
                '互助支援行为': 0
            }
            
            # 基于关键词计算行为得分
            for keyword in keywords:
                keyword_lower = keyword.lower()
                for behavior_type, categories in self.behavior_keywords.items():
                    for category, behavior_keywords_list in categories.items():
                        for behavior_keyword in behavior_keywords_list:
                            if behavior_keyword in keyword_lower:
                                behavior_scores[behavior_type] += 1
            
            # 记录得分最高的行为类型
            if max(behavior_scores.values()) > 0:
                dominant_behavior = max(behavior_scores.items(), key=lambda x: x[1])[0]
                topic_behavior_mapping[topic_id] = {
                    'behavior': dominant_behavior,
                    'scores': behavior_scores,
                    'keywords': keywords
                }
            else:
                # 如果没有明显的行为特征，默认为信息搜索行为
                topic_behavior_mapping[topic_id] = {
                    'behavior': '信息搜索行为',
                    'scores': behavior_scores,
                    'keywords': keywords
                }
        
        self.topic_behavior_mapping = topic_behavior_mapping
        logger.info(f"主题行为映射完成，共映射 {len(topic_behavior_mapping)} 个主题")
        
        return topic_behavior_mapping
    
    def create_behavior_labels_from_topics(self, topic_results):
        """基于主题分配创建行为标签"""
        logger.info("基于主题分配创建行为标签...")
        
        behavior_labels = []
        
        for _, row in topic_results.iterrows():
            topic_id = row['Topic']
            if topic_id in self.topic_behavior_mapping:
                behavior_labels.append(self.topic_behavior_mapping[topic_id]['behavior'])
            else:
                # 对于未映射的主题，默认为信息搜索行为
                behavior_labels.append('信息搜索行为')
        
        return behavior_labels
    
    def train_behavior_classifier(self, texts, behavior_labels):
        """训练行为分类器"""
        logger.info("训练行为分类器...")
        
        # 使用TF-IDF特征
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        X_tfidf = self.vectorizer.fit_transform(texts)
        
        # 训练随机森林分类器
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            criterion='gini',
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        
        # 训练模型
        self.classifier.fit(X_tfidf, behavior_labels)
        
        # 预测训练集
        y_pred = self.classifier.predict(X_tfidf)
        
        # 计算评估指标
        precision, recall, f1, support = precision_recall_fscore_support(behavior_labels, y_pred, average='weighted')
        
        logger.info(f"分类器训练完成")
        logger.info(f"精确率: {precision:.4f}")
        logger.info(f"召回率: {recall:.4f}")
        logger.info(f"F1分数: {f1:.4f}")
        
        # 生成详细报告
        report = classification_report(behavior_labels, y_pred, target_names=['紧急求助行为', '信息搜索行为', '互助支援行为'])
        logger.info(f"分类报告:\n{report}")
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'classification_report': report
        }
    
    def analyze_behavior_distribution(self, behavior_labels):
        """分析行为分布"""
        logger.info("分析行为分布...")
        
        # 统计行为分布
        behavior_counts = pd.Series(behavior_labels).value_counts()
        
        # 创建可视化
        plt.figure(figsize=(15, 10))
        
        # 行为分布饼图
        plt.subplot(2, 3, 1)
        plt.pie(behavior_counts.values, labels=behavior_counts.index, autopct='%1.1f%%')
        plt.title('行为类别分布')
        
        # 行为分布柱状图
        plt.subplot(2, 3, 2)
        behavior_counts.plot(kind='bar')
        plt.title('行为类别数量分布')
        plt.xlabel('行为类别')
        plt.ylabel('数量')
        plt.xticks(rotation=45)
        
        # 主题-行为映射统计
        plt.subplot(2, 3, 3)
        topic_behavior_counts = defaultdict(int)
        for mapping in self.topic_behavior_mapping.values():
            topic_behavior_counts[mapping['behavior']] += 1
        
        topic_behavior_series = pd.Series(topic_behavior_counts)
        topic_behavior_series.plot(kind='bar')
        plt.title('主题-行为映射分布')
        plt.xlabel('行为类别')
        plt.ylabel('主题数量')
        plt.xticks(rotation=45)
        
        # 主题-行为得分热力图
        plt.subplot(2, 3, 4)
        topic_behavior_matrix = []
        topic_ids = []
        behavior_types = ['紧急求助行为', '信息搜索行为', '互助支援行为']
        
        for topic_id, mapping in self.topic_behavior_mapping.items():
            topic_ids.append(f'Topic_{topic_id}')
            scores = [mapping['scores'].get(bt, 0) for bt in behavior_types]
            topic_behavior_matrix.append(scores)
        
        if topic_behavior_matrix:
            sns.heatmap(topic_behavior_matrix, 
                       xticklabels=behavior_types,
                       yticklabels=topic_ids,
                       annot=True, fmt='d', cmap='YlOrRd')
            plt.title('主题-行为得分热力图')
            plt.xlabel('行为类别')
            plt.ylabel('主题ID')
        
        # 行为类别时间分布（如果有时间信息）
        plt.subplot(2, 3, 5)
        behavior_counts.plot(kind='pie', autopct='%1.1f%%')
        plt.title('行为类别占比')
        
        plt.tight_layout()
        
        # 保存图片
        os.makedirs('results/behavior_analysis', exist_ok=True)
        plt.savefig('results/behavior_analysis/existing_topics_behavior_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return behavior_counts
    
    def generate_behavior_report(self, df, behavior_labels, evaluation_results):
        """生成行为分析报告"""
        logger.info("生成行为分析报告...")
        
        # 创建结果DataFrame
        result_df = df.copy()
        result_df['behavior_label'] = behavior_labels
        
        # 保存结果
        os.makedirs('results/behavior_analysis', exist_ok=True)
        result_df.to_csv('results/behavior_analysis/existing_topics_behavior_results.csv', index=False, encoding='utf-8')
        
        # 生成报告
        report = f"""
# 基于现有BERTopic主题的行为分类分析报告

## 1. 实验概述
- 数据量: {len(df)} 条微博
- 主题数量: {len(self.topic_behavior_mapping)} 个有效主题
- 行为类别: 3类（紧急求助行为、信息搜索行为、互助支援行为）

## 2. 模型性能
- 精确率: {evaluation_results['precision']:.4f}
- 召回率: {evaluation_results['recall']:.4f}
- F1分数: {evaluation_results['f1']:.4f}

## 3. 行为分布
"""
        
        # 添加行为分布统计
        behavior_counts = pd.Series(behavior_labels).value_counts()
        for behavior, count in behavior_counts.items():
            percentage = (count / len(behavior_labels)) * 100
            report += f"- {behavior}: {count} 条 ({percentage:.1f}%)\n"
        
        report += f"""
## 4. 主题-行为映射详情
"""
        
        for topic_id, mapping in self.topic_behavior_mapping.items():
            report += f"- Topic {topic_id}: {mapping['behavior']}\n"
            report += f"  关键词: {', '.join(mapping['keywords'])}\n"
            report += f"  行为得分: {mapping['scores']}\n"
        
        report += f"""
## 5. 技术方法
- 主题建模: 基于现有BERTopic结果
- 行为映射: 关键词匹配
- 分类器: RandomForest
- 特征提取: TF-IDF

## 6. 文件输出
- 分类结果: results/behavior_analysis/existing_topics_behavior_results.csv
- 可视化图表: results/behavior_analysis/existing_topics_behavior_analysis.png
- 模型文件: results/behavior_analysis/existing_topics_behavior_model.pkl
"""
        
        # 保存报告
        with open('results/behavior_analysis/existing_topics_behavior_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 保存模型
        model_data = {
            'classifier': self.classifier,
            'vectorizer': self.vectorizer,
            'topic_behavior_mapping': self.topic_behavior_mapping,
            'behavior_keywords': self.behavior_keywords
        }
        
        with open('results/behavior_analysis/existing_topics_behavior_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info("行为分析报告生成完成")
        return report
    
    def run_behavior_classification(self, topic_results_path, topic_info_path, original_data_path):
        """运行基于现有主题的行为分类"""
        logger.info("开始基于现有主题的行为分类...")
        
        # 1. 加载现有主题建模结果
        topic_results, topic_info = self.load_existing_results(topic_results_path, topic_info_path)
        if topic_results is None:
            return None
        
        # 2. 加载原始数据
        df = pd.read_csv(original_data_path)
        texts = df['txt'].fillna('').tolist()
        
        # 3. 分析主题关键词，映射到行为类别
        topic_behavior_mapping = self.analyze_topic_keywords_for_behavior(topic_info, texts)
        
        # 4. 基于主题分配创建行为标签
        behavior_labels = self.create_behavior_labels_from_topics(topic_results)
        
        # 5. 训练行为分类器
        evaluation_results = self.train_behavior_classifier(texts, behavior_labels)
        
        # 6. 分析行为分布
        behavior_distribution = self.analyze_behavior_distribution(behavior_labels)
        
        # 7. 生成报告
        report = self.generate_behavior_report(df, behavior_labels, evaluation_results)
        
        logger.info("基于现有主题的行为分类完成")
        return {
            'df': df,
            'behavior_labels': behavior_labels,
            'evaluation_results': evaluation_results,
            'behavior_distribution': behavior_distribution,
            'report': report
        }

def main():
    """主函数"""
    # 初始化分类器
    classifier = ExistingTopicsBehaviorClassifier()
    
    # 文件路径（根据实际情况调整）
    topic_results_path = '../weibo/results/5_4/topic_modeling_results.csv'
    topic_info_path = '../weibo/results/5_4/topic_info.csv'
    original_data_path = '../weibo/data/weibo_clean_data.csv'
    
    # 检查文件是否存在
    if not os.path.exists(topic_results_path):
        logger.error(f"主题建模结果文件不存在: {topic_results_path}")
        return
    
    if not os.path.exists(topic_info_path):
        logger.error(f"主题信息文件不存在: {topic_info_path}")
        return
    
    if not os.path.exists(original_data_path):
        logger.error(f"原始数据文件不存在: {original_data_path}")
        return
    
    # 运行行为分类
    results = classifier.run_behavior_classification(topic_results_path, topic_info_path, original_data_path)
    
    if results:
        logger.info("基于现有主题的行为分类成功完成！")
    else:
        logger.error("基于现有主题的行为分类失败")

if __name__ == "__main__":
    main() 