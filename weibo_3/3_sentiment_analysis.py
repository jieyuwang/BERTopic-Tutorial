#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3. 情感分析脚本
基于改进的情感分类体系，配合新的行为分类
"""

import pandas as pd
import numpy as np
import jieba
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report
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
    
    logger = logging.getLogger('sentiment_analysis')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = f'data/log/3_sentiment_analysis_{timestamp}.log'
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

class SentimentAnalyzer:
    def __init__(self):
        """初始化情感分析器"""
        logger.info("初始化情感分析器...")
        self.sentiment_keywords = self._build_sentiment_keywords()
        self.model = None
        self.vectorizer = None
        
    def _build_sentiment_keywords(self):
        """构建改进的情感关键词词典"""
        sentiment_keywords = {
            '积极情感': {
                '希望乐观': ['希望', '相信', '期待', '乐观', '信心', '光明', '未来', '好转'],
                '感激温暖': ['感谢', '感激', '感动', '温暖', '温馨', '贴心', '暖心', '善良'],
                '支持鼓励': ['支持', '鼓励', '加油', '坚持', '坚强', '勇敢', '努力', '奋斗'],
                '团结互助': ['团结', '互助', '合作', '一起', '共同', '携手', '同心', '协力']
            },
            '消极情感': {
                '恐惧担忧': ['害怕', '恐惧', '担心', '担忧', '焦虑', '紧张', '不安', '恐慌'],
                '愤怒不满': ['愤怒', '气愤', '恼火', '不满', '抱怨', '抗议', '谴责', '愤怒'],
                '悲伤绝望': ['悲伤', '难过', '痛苦', '绝望', '无助', '无奈', '失望', '沮丧'],
                '孤独无助': ['孤独', '无助', '孤立', '无援', '寂寞', '孤单', '被遗忘', '被忽视']
            },
            '中性情感': {
                '客观描述': ['报道', '通知', '公告', '发布', '通报', '说明', '介绍', '描述'],
                '理性分析': ['分析', '思考', '考虑', '评估', '判断', '理性', '客观', '冷静'],
                '信息传递': ['转发', '分享', '传播', '扩散', '告知', '提醒', '通知', '传达'],
                '日常表达': ['日常', '正常', '普通', '一般', '平常', '习惯', '规律', '稳定']
            }
        }
        return sentiment_keywords
    
    def segment_text(self, text):
        """文本分词"""
        if pd.isna(text):
            return []
        words = jieba.lcut(str(text))
        return [word for word in words if word.strip()]
    
    def extract_sentiment_features(self, text):
        """提取情感特征"""
        words = self.segment_text(text)
        text_lower = str(text).lower()
        
        # 初始化特征向量
        features = {
            'positive_sentiment': 0,
            'negative_sentiment': 0,
            'neutral_sentiment': 0
        }
        
        # 计算各类情感的匹配度
        for sentiment_type, categories in self.sentiment_keywords.items():
            score = 0
            for category, keywords in categories.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        score += 1
            
            if '积极' in sentiment_type:
                features['positive_sentiment'] = score
            elif '消极' in sentiment_type:
                features['negative_sentiment'] = score
            elif '中性' in sentiment_type:
                features['neutral_sentiment'] = score
        
        return features
    
    def create_training_data(self, df, sample_size=3000):
        """创建训练数据"""
        logger.info("创建情感分析训练数据...")
        
        # 随机抽样
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        # 基于关键词规则进行标注
        def label_sentiment(text):
            text_lower = str(text).lower()
            
            # 计算各类情感的匹配度
            scores = {
                '积极情感': 0,
                '消极情感': 0,
                '中性情感': 0
            }
            
            for sentiment_type, categories in self.sentiment_keywords.items():
                for category, keywords in categories.items():
                    for keyword in keywords:
                        if keyword in text_lower:
                            scores[sentiment_type] += 1
            
            # 返回得分最高的情感类型
            if max(scores.values()) == 0:
                return '中性情感'  # 默认分类
            else:
                return max(scores.items(), key=lambda x: x[1])[0]
        
        sample_df['sentiment_label'] = sample_df['txt'].apply(label_sentiment)
        
        return sample_df
    
    def train_sentiment_model(self, training_data):
        """训练情感分析模型"""
        logger.info("训练情感分析模型...")
        
        # 准备训练数据
        X = training_data['txt'].fillna('')
        y = training_data['sentiment_label']
        
        # 使用TF-IDF特征提取
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        X_tfidf = self.vectorizer.fit_transform(X)
        
        # 训练逻辑回归模型
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
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
        
        logger.info(f"情感分析模型评估结果:")
        logger.info(f"准确率: {precision:.3f}")
        logger.info(f"召回率: {recall:.3f}")
        logger.info(f"F1分数: {f1:.3f}")
        
        # 详细分类报告
        logger.info("\n详细分类报告:")
        logger.info(classification_report(y_test, y_pred))
        
        return evaluation_results, X_test, y_test, y_pred
    
    def predict_sentiments(self, df):
        """预测整个数据集的情感类型"""
        logger.info("预测情感类型...")
        
        if self.model is None or self.vectorizer is None:
            raise ValueError("模型未训练，请先调用train_sentiment_model方法")
        
        # 特征提取
        X = df['txt'].fillna('')
        X_tfidf = self.vectorizer.transform(X)
        
        # 预测
        predictions = self.model.predict(X_tfidf)
        probabilities = self.model.predict_proba(X_tfidf)
        
        # 添加预测结果到数据框
        df_with_sentiment = df.copy()
        df_with_sentiment['sentiment_type'] = predictions
        df_with_sentiment['sentiment_confidence'] = np.max(probabilities, axis=1)
        
        # 计算情感得分（积极情感为正，消极情感为负，中性情感为0）
        sentiment_scores = []
        for pred, prob in zip(predictions, probabilities):
            if pred == '积极情感':
                score = prob[list(self.model.classes_).index('积极情感')]
            elif pred == '消极情感':
                score = -prob[list(self.model.classes_).index('消极情感')]
            else:
                score = 0
            sentiment_scores.append(score)
        
        df_with_sentiment['sentiment_score'] = sentiment_scores
        
        # 添加各类情感的概率
        sentiment_classes = self.model.classes_
        for i, sentiment in enumerate(sentiment_classes):
            df_with_sentiment[f'{sentiment}_probability'] = probabilities[:, i]
        
        return df_with_sentiment
    
    def analyze_sentiment_distribution(self, df_with_sentiment):
        """分析情感分布"""
        logger.info("分析情感分布...")
        
        # 情感类型分布
        sentiment_counts = df_with_sentiment['sentiment_type'].value_counts()
        
        plt.figure(figsize=(15, 10))
        
        # 子图1: 情感类型分布饼图
        plt.subplot(2, 3, 1)
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
        plt.title('情感类型分布')
        
        # 子图2: 情感类型分布柱状图
        plt.subplot(2, 3, 2)
        sentiment_counts.plot(kind='bar')
        plt.title('情感类型分布')
        plt.xticks(rotation=45)
        
        # 子图3: 用户类型与情感类型的关系
        plt.subplot(2, 3, 3)
        cross_tab = pd.crosstab(df_with_sentiment['user_type'], df_with_sentiment['sentiment_type'])
        cross_tab.plot(kind='bar', stacked=True)
        plt.title('用户类型与情感类型关系')
        plt.xticks(rotation=45)
        plt.legend(title='情感类型', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 子图4: 情感得分分布
        plt.subplot(2, 3, 4)
        plt.hist(df_with_sentiment['sentiment_score'], bins=30, alpha=0.7, color='skyblue')
        plt.title('情感得分分布')
        plt.xlabel('情感得分')
        plt.ylabel('频次')
        
        # 子图5: 情感置信度分布
        plt.subplot(2, 3, 5)
        plt.hist(df_with_sentiment['sentiment_confidence'], bins=20, alpha=0.7, color='lightgreen')
        plt.title('情感识别置信度分布')
        plt.xlabel('置信度')
        plt.ylabel('频次')
        
        # 子图6: 各类情感概率分布
        plt.subplot(2, 3, 6)
        sentiment_prob_cols = [col for col in df_with_sentiment.columns if col.endswith('_probability')]
        if sentiment_prob_cols:
            prob_data = df_with_sentiment[sentiment_prob_cols]
            prob_data.boxplot()
            plt.title('各类情感概率分布')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('result/sentiment/sentiment_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存情感分析结果
        df_with_sentiment.to_csv('result/sentiment/sentiment_analysis_results.csv', 
                                index=False, encoding='utf-8')
        
        # 生成情感分析报告
        self.generate_sentiment_report(df_with_sentiment)
    
    def generate_sentiment_report(self, df_with_sentiment):
        """生成情感分析报告"""
        sentiment_counts = df_with_sentiment['sentiment_type'].value_counts()
        
        report = f"""
情感分析报告
============

总体统计:
- 总文本数: {len(df_with_sentiment)}
- 情感类型分布:
{'-' * 40}
"""
        
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df_with_sentiment)) * 100
            report += f"{sentiment}: {count} ({percentage:.1f}%)\n"
        
        report += f"""
详细分析:
{'-' * 40}

1. 积极情感:
- 数量: {len(df_with_sentiment[df_with_sentiment['sentiment_type'] == '积极情感'])}
- 平均置信度: {df_with_sentiment[df_with_sentiment['sentiment_type'] == '积极情感']['sentiment_confidence'].mean():.3f}
- 平均得分: {df_with_sentiment[df_with_sentiment['sentiment_type'] == '积极情感']['sentiment_score'].mean():.3f}

2. 消极情感:
- 数量: {len(df_with_sentiment[df_with_sentiment['sentiment_type'] == '消极情感'])}
- 平均置信度: {df_with_sentiment[df_with_sentiment['sentiment_type'] == '消极情感']['sentiment_confidence'].mean():.3f}
- 平均得分: {df_with_sentiment[df_with_sentiment['sentiment_type'] == '消极情感']['sentiment_score'].mean():.3f}

3. 中性情感:
- 数量: {len(df_with_sentiment[df_with_sentiment['sentiment_type'] == '中性情感'])}
- 平均置信度: {df_with_sentiment[df_with_sentiment['sentiment_type'] == '中性情感']['sentiment_confidence'].mean():.3f}
- 平均得分: {df_with_sentiment[df_with_sentiment['sentiment_type'] == '中性情感']['sentiment_score'].mean():.3f}

用户类型情感分析:
{'-' * 40}
"""
        
        user_sentiment = pd.crosstab(df_with_sentiment['user_type'], df_with_sentiment['sentiment_type'])
        report += user_sentiment.to_string()
        
        report += f"""

情感得分统计:
{'-' * 40}
- 平均情感得分: {df_with_sentiment['sentiment_score'].mean():.3f}
- 情感得分标准差: {df_with_sentiment['sentiment_score'].std():.3f}
- 最高情感得分: {df_with_sentiment['sentiment_score'].max():.3f}
- 最低情感得分: {df_with_sentiment['sentiment_score'].min():.3f}
- 积极情感比例: {len(df_with_sentiment[df_with_sentiment['sentiment_score'] > 0]) / len(df_with_sentiment) * 100:.1f}%
- 消极情感比例: {len(df_with_sentiment[df_with_sentiment['sentiment_score'] < 0]) / len(df_with_sentiment) * 100:.1f}%
- 中性情感比例: {len(df_with_sentiment[df_with_sentiment['sentiment_score'] == 0]) / len(df_with_sentiment) * 100:.1f}%
"""
        
        with open('result/sentiment/sentiment_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info("情感分析报告已保存到 result/sentiment/sentiment_analysis_report.txt")

def main():
    """主函数"""
    logger.info("=== 3. 情感分析开始 ===")
    
    # 加载数据
    try:
        df = pd.read_csv('data/clean/processed_data.csv')
        logger.info("加载清洗后数据")
    except:
        logger.info("数据加载失败，请先运行数据预处理脚本")
        return
    
    logger.info(f"数据规模: {len(df)} 条记录")
    
    # 初始化情感分析器
    analyzer = SentimentAnalyzer()
    
    # 创建训练数据
    training_data = analyzer.create_training_data(df, sample_size=3000)
    
    # 训练模型
    evaluation_results, X_test, y_test, y_pred = analyzer.train_sentiment_model(training_data)
    
    # 预测整个数据集
    df_with_sentiment = analyzer.predict_sentiments(df)
    
    # 分析情感分布
    analyzer.analyze_sentiment_distribution(df_with_sentiment)
    
    # 保存评估结果
    evaluation_df = pd.DataFrame([evaluation_results])
    evaluation_df.to_csv('result/sentiment/sentiment_evaluation.csv', index=False)
    
    logger.info("=== 3. 情感分析完成 ===")
    return df_with_sentiment, evaluation_results

if __name__ == "__main__":
    main() 