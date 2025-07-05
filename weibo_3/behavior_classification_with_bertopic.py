#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于BERTopic的行为分类脚本
结合主题建模和行为关键词词典，实现高质量的行为分类
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
import warnings
import logging
import os
from datetime import datetime
import pickle
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
def setup_logger():
    """设置日志记录器"""
    os.makedirs('data/log', exist_ok=True)
    
    logger = logging.getLogger('bertopic_behavior_classification')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = f'data/log/bertopic_behavior_classification_{timestamp}.log'
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

class BERTopicBehaviorClassifier:
    def __init__(self):
        """初始化BERTopic行为分类器"""
        self.behavior_keywords = self._build_behavior_keywords()
        self.topic_model = None
        self.embedding_model = None
        self.topic_behavior_mapping = {}
        self.classifier = None
        self.vectorizer = None
        
    def _build_behavior_keywords(self):
        """构建行为关键词词典"""
        logger.info("构建行为关键词词典...")
        behavior_keywords = {
            '紧急求助行为': {
                '生命安全求助': ['救命', '被困', '危险', '紧急', '快', '急', '救援', '转移', '疏散', '洪水', '台风', '暴雨', '灾害', '险情'],
                '生活需求求助': ['求助', '需要帮助', '帮忙', '支持', '援助', '食物', '水', '药品', '物资', '救援'],
                '信息需求求助': ['请问', '谁知道', '求问', '咨询', '了解', '情况', '最新', '怎么样', '如何']
            },
            '信息搜索行为': {
                '主动询问': ['请问', '谁知道', '求问', '帮忙', '需要', '了解', '什么', '哪里', '什么时候', '怎么'],
                '信息求证': ['真的吗', '确认', '核实', '求证', '是否', '真假', '消息', '准确', '可靠'],
                '关注动态': ['关注', '跟踪', '更新', '最新', '情况', '进展', '新闻', '报道', '通知'],
                '寻求建议': ['建议', '怎么办', '如何', '应该', '推荐', '意见', '方法', '处理']
            },
            '互助支援行为': {
                '紧急救援': ['救援', '抢险', '救助', '转移', '疏散', '志愿者', '参与', '奔赴', '支援', '协助'],
                '物资支援': ['捐赠', '支援', '提供', '分享', '免费', '物资', '帮助', '捐献', '贡献', '爱心'],
                '志愿服务': ['志愿者', '互助', '团结', '一起', '参与', '服务', '义务', '公益', '志愿'],
                '情感支持': ['加油', '支持', '鼓励', '安慰', '理解', '坚强', '挺住', '坚持', '相信', '希望'],
                '组织协调': ['组织', '协调', '安排', '调度', '指挥', '统筹', '配合', '合作', '联合'],
                '信息传播': ['转发', '扩散', '传播', '分享', '告知', '通知', '提醒', '预警']
            }
        }
        logger.info(f"行为关键词词典构建完成，包含 {sum(len(cat) for behavior in behavior_keywords.values() for cat in behavior.values())} 个关键词")
        return behavior_keywords
    
    def load_data(self, data_path, max_samples=1000):
        """加载数据"""
        logger.info(f"加载数据: {data_path}")
        try:
            df = pd.read_csv(data_path)
            # 固定采样策略：始终使用前max_samples条数据，确保结果可重现
            if len(df) > max_samples:
                logger.info(f"数据量过大({len(df)})，取前{max_samples}条进行测试...")
                df = df.head(max_samples)
            logger.info(f"数据加载成功，共 {len(df)} 条记录")
            return df
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            return None
    
    def preprocess_text(self, text):
        """文本预处理"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        # 去除特殊字符，保留中文、英文、数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def train_bertopic_model(self, texts, embedding_model_name='paraphrase-multilingual-MiniLM-L12-v2', sample_size=1000):
        """训练BERTopic模型"""
        logger.info("开始训练BERTopic模型...")
        
        # 固定采样策略：使用前sample_size条数据
        if len(texts) > sample_size:
            logger.info(f"数据量过大({len(texts)})，取前{sample_size}条进行训练...")
            sample_texts = texts[:sample_size]
        else:
            sample_texts = texts
        
        # 初始化嵌入模型
        self.embedding_model = SentenceTransformer(embedding_model_name)
        logger.info(f"使用嵌入模型: {embedding_model_name}")
        
        # 生成嵌入
        logger.info("生成文本嵌入...")
        embeddings = self.embedding_model.encode(sample_texts, show_progress_bar=True)
        
        # 训练BERTopic模型（优化参数设置）
        try:
            self.topic_model = BERTopic(
                nr_topics=15,  # 固定主题数量，避免随机性
                min_topic_size=15,  # 增加最小主题大小，提高质量
                verbose=True,
                calculate_probabilities=True,
                random_state=42  # 固定随机种子
            )
            
            topics, probs = self.topic_model.fit_transform(sample_texts, embeddings)
        except ImportError as e:
            logger.warning(f"BERTopic版本兼容性问题: {e}")
            logger.info("尝试使用备用方法...")
            
            # 备用方法：直接使用嵌入进行聚类
            from sklearn.cluster import HDBSCAN
            from sklearn.manifold import UMAP
            
            # UMAP降维
            umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
            umap_embeddings = umap_model.fit_transform(embeddings)
            
            # HDBSCAN聚类
            hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', random_state=42)
            topics = hdbscan_model.fit_predict(umap_embeddings)
            
            # 创建伪BERTopic对象
            class PseudoBERTopic:
                def __init__(self, topics, embeddings, texts):
                    self.topics_ = topics
                    self.embeddings_ = embeddings
                    self.texts_ = texts
                
                def get_topic_info(self):
                    topic_counts = pd.Series(topics).value_counts().reset_index()
                    topic_counts.columns = ['Topic', 'Count']
                    return topic_counts
                
                def get_topics(self):
                    # 简化版本，返回主题ID和关键词
                    topics_dict = {}
                    for topic_id in set(topics):
                        if topic_id != -1:
                            topics_dict[topic_id] = [f"topic_{topic_id}"]
                    return topics_dict
            
            self.topic_model = PseudoBERTopic(topics, embeddings, sample_texts)
            probs = None
        
        logger.info(f"BERTopic模型训练完成，发现 {len(self.topic_model.get_topics())} 个主题")
        
        return topics, probs
    
    def map_topics_to_behaviors(self, texts, topics):
        """将主题映射到行为类别"""
        logger.info("开始主题到行为的映射...")
        
        # 确保texts和topics长度匹配
        if len(texts) != len(topics):
            logger.warning(f"texts长度({len(texts)})与topics长度({len(topics)})不匹配，使用较短的长度")
            min_length = min(len(texts), len(topics))
            texts = texts[:min_length]
            topics = topics[:min_length]
        
        # 获取主题信息
        topic_info = self.topic_model.get_topic_info()
        topic_keywords = self.topic_model.get_topics()
        
        # 为每个主题计算行为得分
        topic_behavior_scores = {}
        
        for topic_id in topic_info['Topic'].unique():
            if topic_id == -1:  # 跳过异常值主题
                continue
                
            # 获取该主题的文档
            topic_docs = [texts[i] for i in range(len(texts)) if topics[i] == topic_id]
            if len(topic_docs) == 0:
                continue
            
            # 计算该主题的行为得分
            behavior_scores = {
                '紧急求助行为': 0,
                '信息搜索行为': 0,
                '互助支援行为': 0
            }
            
            # 改进的关键词匹配逻辑
            for doc in topic_docs:
                doc_lower = str(doc).lower()
                
                # 紧急求助行为关键词匹配
                for category, keywords in self.behavior_keywords['紧急求助行为'].items():
                    for keyword in keywords:
                        if keyword in doc_lower:
                            behavior_scores['紧急求助行为'] += 1
                
                # 信息搜索行为关键词匹配
                for category, keywords in self.behavior_keywords['信息搜索行为'].items():
                    for keyword in keywords:
                        if keyword in doc_lower:
                            behavior_scores['信息搜索行为'] += 1
                
                # 互助支援行为关键词匹配
                for category, keywords in self.behavior_keywords['互助支援行为'].items():
                    for keyword in keywords:
                        if keyword in doc_lower:
                            behavior_scores['互助支援行为'] += 1
                
                # 移除额外的语义分析加分，避免过度偏向互助支援行为
                # 只使用预定义的关键词词典进行匹配
            
            # 记录得分最高的行为类型
            if max(behavior_scores.values()) > 0:
                # 平衡的分类策略：选择得分最高的行为类型，但避免过度偏向
                support_score = behavior_scores['互助支援行为']
                emergency_score = behavior_scores['紧急求助行为']
                info_score = behavior_scores['信息搜索行为']
                
                # 如果紧急求助行为得分明显高于其他行为，优先分类为紧急求助行为
                if emergency_score > support_score * 1.5 and emergency_score > info_score * 1.5:
                    dominant_behavior = '紧急求助行为'
                # 如果信息搜索行为得分明显高于其他行为，优先分类为信息搜索行为
                elif info_score > support_score * 1.5 and info_score > emergency_score * 1.5:
                    dominant_behavior = '信息搜索行为'
                # 否则选择得分最高的行为类型
                else:
                    dominant_behavior = max(behavior_scores.items(), key=lambda x: x[1])[0]
                
                topic_behavior_scores[topic_id] = {
                    'behavior': dominant_behavior,
                    'scores': behavior_scores,
                    'doc_count': len(topic_docs)
                }
                
                # 记录详细的得分信息
                logger.info(f"Topic {topic_id}: {dominant_behavior} (得分: {behavior_scores})")
        
        self.topic_behavior_mapping = topic_behavior_scores
        logger.info(f"主题行为映射完成，共映射 {len(topic_behavior_scores)} 个主题")
        
        # 统计行为分布
        behavior_distribution = {}
        for mapping in topic_behavior_scores.values():
            behavior = mapping['behavior']
            behavior_distribution[behavior] = behavior_distribution.get(behavior, 0) + mapping['doc_count']
        
        logger.info(f"行为分布: {behavior_distribution}")
        
        return topic_behavior_scores
    
    def create_behavior_labels(self, topics):
        """基于主题映射创建行为标签"""
        logger.info("创建行为标签...")
        
        behavior_labels = []
        for topic_id in topics:
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
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, behavior_labels, test_size=0.2, random_state=42, stratify=behavior_labels
        )
        
        # 训练模型
        self.classifier.fit(X_train, y_train)
        
        # 评估模型
        y_pred = self.classifier.predict(X_test)
        
        # 计算评估指标
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        logger.info(f"分类器训练完成")
        logger.info(f"精确率: {precision:.4f}")
        logger.info(f"召回率: {recall:.4f}")
        logger.info(f"F1分数: {f1:.4f}")
        
        # 生成详细报告
        unique_labels = sorted(set(list(y_test) + list(y_pred)))
        target_names = []
        for label in unique_labels:
            if '紧急求助' in label:
                target_names.append('紧急求助行为')
            elif '信息搜索' in label:
                target_names.append('信息搜索行为')
            elif '互助支援' in label:
                target_names.append('互助支援行为')
            else:
                target_names.append(label)
        
        report = classification_report(y_test, y_pred, target_names=target_names)
        logger.info(f"分类报告:\n{report}")
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'classification_report': report
        }
    
    def predict_behaviors(self, texts):
        """预测行为类别"""
        if self.classifier is None or self.vectorizer is None:
            logger.error("分类器未训练，请先调用train_behavior_classifier")
            return None
        
        # 特征提取
        X_tfidf = self.vectorizer.transform(texts)
        
        # 预测
        predictions = self.classifier.predict(X_tfidf)
        probabilities = self.classifier.predict_proba(X_tfidf)
        
        return predictions, probabilities
    
    def analyze_behavior_distribution(self, behavior_labels):
        """分析行为分布"""
        logger.info("分析行为分布...")
        
        # 统计行为分布
        behavior_counts = pd.Series(behavior_labels).value_counts()
        
        # 创建可视化
        plt.figure(figsize=(12, 8))
        
        # 行为分布饼图
        plt.subplot(2, 2, 1)
        plt.pie(behavior_counts.values, labels=behavior_counts.index, autopct='%1.1f%%')
        plt.title('行为类别分布')
        
        # 行为分布柱状图
        plt.subplot(2, 2, 2)
        behavior_counts.plot(kind='bar')
        plt.title('行为类别数量分布')
        plt.xlabel('行为类别')
        plt.ylabel('数量')
        plt.xticks(rotation=45)
        
        # 主题-行为映射热力图
        plt.subplot(2, 2, 3)
        topic_behavior_matrix = []
        topic_ids = []
        behavior_types = ['紧急求助行为', '信息搜索行为', '互助支援行为']
        
        for topic_id, mapping in self.topic_behavior_mapping.items():
            topic_ids.append(f'Topic_{topic_id}')
            scores = [mapping['scores'].get(bt, 0) for bt in behavior_types]
            topic_behavior_matrix.append(scores)
        
        if topic_behavior_matrix:
            # 将浮点数转换为整数用于热力图显示
            topic_behavior_matrix_int = [[int(score) for score in row] for row in topic_behavior_matrix]
            sns.heatmap(topic_behavior_matrix_int, 
                       xticklabels=behavior_types,
                       yticklabels=topic_ids,
                       annot=True, fmt='d', cmap='YlOrRd')
            plt.title('主题-行为得分热力图')
            plt.xlabel('行为类别')
            plt.ylabel('主题ID')
        
        plt.tight_layout()
        
        # 保存图片
        os.makedirs('result/behavior_analysis', exist_ok=True)
        plt.savefig('result/behavior_analysis/behavior_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return behavior_counts
    
    def generate_behavior_report(self, df, behavior_labels, evaluation_results):
        """生成行为分析报告"""
        logger.info("生成行为分析报告...")
        
        # 创建结果DataFrame
        result_df = df.copy()
        result_df['behavior_label'] = behavior_labels
        
        # 保存结果
        os.makedirs('result/behavior_analysis', exist_ok=True)
        result_df.to_csv('result/behavior_analysis/behavior_classification_results.csv', index=False, encoding='utf-8')
        
        # 生成报告
        report = f"""
# BERTopic行为分类分析报告

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
## 4. 主题-行为映射
"""
        
        for topic_id, mapping in self.topic_behavior_mapping.items():
            report += f"- Topic {topic_id}: {mapping['behavior']} ({mapping['doc_count']} 条文档)\n"
        
        report += f"""
## 5. 技术方法
- 嵌入模型: SentenceTransformer
- 主题建模: BERTopic
- 分类器: RandomForest
- 特征提取: TF-IDF

## 6. 文件输出
- 分类结果: result/behavior_analysis/behavior_classification_results.csv
- 可视化图表: result/behavior_analysis/behavior_distribution_analysis.png
- 模型文件: result/behavior_analysis/bertopic_behavior_model.pkl
"""
        
        # 保存报告
        with open('result/behavior_analysis/behavior_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 保存模型
        model_data = {
            'topic_model': self.topic_model,
            'embedding_model': self.embedding_model,
            'classifier': self.classifier,
            'vectorizer': self.vectorizer,
            'topic_behavior_mapping': self.topic_behavior_mapping,
            'behavior_keywords': self.behavior_keywords
        }
        
        with open('result/behavior_analysis/bertopic_behavior_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info("行为分析报告生成完成")
        return report
    
    def run_full_pipeline(self, data_path):
        """运行完整的行为分类流程"""
        logger.info("开始运行BERTopic行为分类完整流程...")
        
        # 1. 加载数据
        df = self.load_data(data_path)
        if df is None:
            return None
        
        # 2. 数据预处理
        logger.info("数据预处理...")
        df['processed_text'] = df['txt'].apply(self.preprocess_text)
        texts = df['processed_text'].tolist()
        
        # 3. 训练BERTopic模型
        topics, probs = self.train_bertopic_model(texts)
        
        # 4. 主题到行为映射
        topic_behavior_mapping = self.map_topics_to_behaviors(texts, topics)
        
        # 5. 创建行为标签
        behavior_labels = self.create_behavior_labels(topics)
        
        # 6. 训练行为分类器
        evaluation_results = self.train_behavior_classifier(texts, behavior_labels)
        
        # 7. 分析行为分布
        behavior_distribution = self.analyze_behavior_distribution(behavior_labels)
        
        # 8. 生成报告
        report = self.generate_behavior_report(df, behavior_labels, evaluation_results)
        
        # 9. 详细分析（从测试脚本合并的功能）
        self._detailed_analysis(texts, behavior_labels, topic_behavior_mapping, behavior_distribution)
        
        logger.info("BERTopic行为分类完整流程执行完成")
        return {
            'df': df,
            'topics': topics,
            'behavior_labels': behavior_labels,
            'evaluation_results': evaluation_results,
            'behavior_distribution': behavior_distribution,
            'report': report
        }
    
    def _detailed_analysis(self, texts, behavior_labels, topic_behavior_mapping, behavior_distribution):
        """详细分析功能（从测试脚本合并）"""
        logger.info("\n=== 详细行为分析 ===")
        for behavior_type in ['紧急求助行为', '信息搜索行为', '互助支援行为']:
            count = behavior_distribution.get(behavior_type, 0)
            percentage = (count / len(behavior_labels)) * 100 if len(behavior_labels) > 0 else 0
            logger.info(f"{behavior_type}: {count} 条 ({percentage:.1f}%)")
        
        # 分析主题-行为映射
        logger.info("\n=== 主题-行为映射分析 ===")
        for topic_id, mapping in topic_behavior_mapping.items():
            logger.info(f"Topic {topic_id}: {mapping['behavior']} (文档数: {mapping['doc_count']})")
            logger.info(f"  得分详情: {mapping['scores']}")
        
        # 检查是否有互助支援行为
        support_behavior_count = behavior_distribution.get('互助支援行为', 0)
        if support_behavior_count > 0:
            logger.info(f"\n✅ 成功识别到 {support_behavior_count} 条互助支援行为")
            
            # 显示互助支援行为的示例
            support_indices = [i for i, label in enumerate(behavior_labels) if label == '互助支援行为']
            logger.info("\n互助支援行为示例:")
            for i, idx in enumerate(support_indices[:3]):  # 显示前3个示例
                logger.info(f"示例 {i+1}: {texts[idx][:100]}...")
        else:
            logger.warning("\n⚠️ 未识别到互助支援行为，可能需要进一步调整关键词或数据")
            
            # 分析可能的原因
            logger.info("\n分析可能的原因:")
            
            # 检查数据中是否包含支援相关词汇
            support_keywords = ['支援', '帮助', '协助', '救援', '救助', '志愿者', '捐赠', '提供', '分享', '免费', '互助', '团结', '一起', '参与', '服务', '加油', '支持', '鼓励', '安慰', '理解', '坚强']
            found_keywords = []
            for keyword in support_keywords:
                count = sum(1 for text in texts if keyword in text.lower())
                if count > 0:
                    found_keywords.append((keyword, count))
            
            if found_keywords:
                logger.info("数据中包含的支援相关词汇:")
                for keyword, count in sorted(found_keywords, key=lambda x: x[1], reverse=True):
                    logger.info(f"  '{keyword}': {count} 次")
            else:
                logger.info("数据中未发现支援相关词汇")
        
        logger.info("\n详细分析完成！")

def main():
    """主函数"""
    # 初始化分类器
    classifier = BERTopicBehaviorClassifier()
    
    # 运行完整流程
    data_path = 'data/clean/processed_data.csv'  # 使用weibo_3目录下的数据
    
    if os.path.exists(data_path):
        results = classifier.run_full_pipeline(data_path)
        if results:
            logger.info("BERTopic行为分类成功完成！")
        else:
            logger.error("BERTopic行为分类失败")
    else:
        logger.error(f"数据文件不存在: {data_path}")

if __name__ == "__main__":
    main() 