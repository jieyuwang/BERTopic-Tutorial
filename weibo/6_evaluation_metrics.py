#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
微博数据BERTopic主题建模评估指标计算脚本
计算主题一致性、多样性、覆盖度等评估指标
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import re
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class Logger:
    """日志记录器，同时输出到控制台和文件"""
    def __init__(self, log_file):
        self.log_file = log_file
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

class TopicModelEvaluator:
    """主题模型评估器"""
    
    def __init__(self, topic_model, embeddings, docs, topic_docs):
        """
        初始化评估器
        
        Args:
            topic_model: BERTopic模型
            embeddings: 文档嵌入向量
            docs: 文档列表
            topic_docs: 主题文档信息DataFrame
        """
        self.topic_model = topic_model
        self.embeddings = embeddings
        self.docs = docs
        self.topic_docs = topic_docs
        self.topic_info = topic_model.get_topic_info()
        
    def calculate_topic_coherence(self, top_n=10):
        """计算主题一致性"""
        print("计算主题一致性...")
        
        coherence_scores = []
        valid_topics = []
        
        for topic_id in self.topic_info['Topic']:
            if topic_id == -1:  # 跳过离群主题
                continue
                
            # 获取主题关键词
            topic_words = self.topic_model.get_topic(topic_id)
            if not topic_words:
                continue
                
            # 提取前top_n个关键词
            words = [word for word, _ in topic_words[:top_n]]
            
            # 计算关键词之间的相似度
            word_embeddings = []
            valid_words = []
            
            for word in words:
                # 找到包含该词的文档（只在前15000个文档中搜索）
                word_docs = [doc for doc in self.docs[:len(self.embeddings)] if word in doc]
                if word_docs:
                    # 计算词的平均嵌入向量
                    word_doc_indices = [i for i, doc in enumerate(self.docs[:len(self.embeddings)]) if word in doc]
                    # 确保索引在有效范围内
                    valid_indices = [i for i in word_doc_indices if i < len(self.embeddings)]
                    if valid_indices:
                        word_embedding = np.mean(self.embeddings[valid_indices], axis=0)
                        word_embeddings.append(word_embedding)
                        valid_words.append(word)
            
            if len(word_embeddings) >= 2:
                # 计算余弦相似度
                similarities = cosine_similarity(word_embeddings)
                # 计算平均相似度（排除对角线）
                avg_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
                coherence_scores.append(avg_similarity)
                valid_topics.append(topic_id)
        
        return coherence_scores, valid_topics
    
    def calculate_topic_diversity(self):
        """计算主题多样性"""
        print("计算主题多样性...")
        
        # 获取所有主题的关键词
        all_words = set()
        topic_word_counts = []
        
        for topic_id in self.topic_info['Topic']:
            if topic_id == -1:
                continue
                
            topic_words = self.topic_model.get_topic(topic_id)
            if topic_words:
                words = [word for word, _ in topic_words[:10]]  # 取前10个关键词
                all_words.update(words)
                topic_word_counts.append(len(words))
        
        # 计算词汇多样性
        total_unique_words = len(all_words)
        total_words = sum(topic_word_counts)
        diversity_score = total_unique_words / total_words if total_words > 0 else 0
        
        return diversity_score, total_unique_words, total_words
    
    def calculate_topic_coverage(self):
        """计算主题覆盖度"""
        print("计算主题覆盖度...")
        
        total_docs = len(self.topic_docs)
        outlier_docs = len(self.topic_docs[self.topic_docs['topic'] == -1])
        covered_docs = total_docs - outlier_docs
        
        coverage_score = covered_docs / total_docs if total_docs > 0 else 0
        
        return coverage_score, covered_docs, total_docs
    
    def calculate_clustering_metrics(self):
        """计算聚类指标"""
        print("计算聚类指标...")
        
        # 过滤掉离群点
        valid_mask = self.topic_docs['topic'] != -1
        valid_embeddings = self.embeddings[valid_mask]
        valid_topics = self.topic_docs.loc[valid_mask, 'topic'].values
        
        if len(valid_topics) < 2:
            return None, None
        
        # 计算轮廓系数
        try:
            silhouette = silhouette_score(valid_embeddings, valid_topics)
        except:
            silhouette = 0
        
        # 计算Calinski-Harabasz指数
        try:
            calinski_harabasz = calinski_harabasz_score(valid_embeddings, valid_topics)
        except:
            calinski_harabasz = 0
        
        return silhouette, calinski_harabasz
    
    def calculate_topic_size_distribution(self):
        """计算主题大小分布"""
        print("计算主题大小分布...")
        
        topic_sizes = self.topic_info[self.topic_info['Topic'] != -1]['Count'].values
        
        if len(topic_sizes) == 0:
            return None
        
        stats = {
            'mean_size': np.mean(topic_sizes),
            'median_size': np.median(topic_sizes),
            'std_size': np.std(topic_sizes),
            'min_size': np.min(topic_sizes),
            'max_size': np.max(topic_sizes),
            'total_topics': len(topic_sizes)
        }
        
        return stats
    
    def calculate_temporal_analysis(self):
        """计算时间分析指标"""
        print("计算时间分析指标...")
        
        if '时间' not in self.topic_docs.columns:
            return None
        
        # 按时间和主题统计
        temporal_stats = {}
        
        for topic_id in self.topic_info['Topic']:
            if topic_id == -1:
                continue
                
            topic_docs = self.topic_docs[self.topic_docs['topic'] == topic_id]
            if len(topic_docs) == 0:
                continue
                
            time_counts = topic_docs['时间'].value_counts()
            temporal_stats[topic_id] = {
                'time_span': len(time_counts),
                'most_active_time': time_counts.index[0],
                'doc_count': len(topic_docs)
            }
        
        return temporal_stats
    
    def generate_evaluation_report(self, output_dir="results"):
        """生成完整的评估报告"""
        print("生成评估报告...")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 计算各项指标
        coherence_scores, valid_topics = self.calculate_topic_coherence()
        diversity_score, unique_words, total_words = self.calculate_topic_diversity()
        coverage_score, covered_docs, total_docs = self.calculate_topic_coverage()
        silhouette, calinski_harabasz = self.calculate_clustering_metrics()
        size_stats = self.calculate_topic_size_distribution()
        temporal_stats = self.calculate_temporal_analysis()
        
        # 汇总评估结果
        evaluation_results = {
            '主题一致性': {
                '平均一致性分数': np.mean(coherence_scores) if coherence_scores else 0,
                '一致性分数标准差': np.std(coherence_scores) if coherence_scores else 0,
                '有效主题数': len(valid_topics)
            },
            '主题多样性': {
                '多样性分数': diversity_score,
                '唯一词汇数': unique_words,
                '总词汇数': total_words
            },
            '主题覆盖度': {
                '覆盖度分数': coverage_score,
                '覆盖文档数': covered_docs,
                '总文档数': total_docs
            },
            '聚类质量': {
                '轮廓系数': silhouette,
                'Calinski-Harabasz指数': calinski_harabasz
            }
        }
        
        if size_stats:
            evaluation_results['主题大小分布'] = size_stats
        
        # 保存评估结果
        results_df = pd.DataFrame()
        for category, metrics in evaluation_results.items():
            for metric, value in metrics.items():
                results_df = pd.concat([results_df, pd.DataFrame({
                    '类别': [category],
                    '指标': [metric],
                    '数值': [value]
                })], ignore_index=True)
        
        results_df.to_csv(os.path.join(output_dir, 'evaluation_metrics_high_quality.csv'), 
                         index=False, encoding='utf-8')
        
        # 保存详细的主题一致性分数
        if coherence_scores:
            coherence_df = pd.DataFrame({
                'Topic': valid_topics,
                'Coherence_Score': coherence_scores
            })
            coherence_df.to_csv(os.path.join(output_dir, 'topic_coherence_high_quality.csv'), 
                               index=False, encoding='utf-8')
        
        # 保存时间分析结果
        if temporal_stats:
            temporal_df = pd.DataFrame(temporal_stats).T.reset_index()
            temporal_df.columns = ['Topic', 'Time_Span', 'Most_Active_Time', 'Doc_Count']
            temporal_df.to_csv(os.path.join(output_dir, 'temporal_analysis_high_quality.csv'), 
                              index=False, encoding='utf-8')
        
        # 打印评估摘要
        print("\n" + "="*60)
        print("BERTopic主题建模评估报告（改进版）")
        print("="*60)
        
        for category, metrics in evaluation_results.items():
            print(f"\n{category}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
        
        # 生成可视化图表
        self.generate_visualizations(output_dir)
        
        return evaluation_results
    
    def generate_visualizations(self, output_dir):
        """生成可视化图表"""
        print("生成可视化图表...")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 主题大小分布图
        if 'Count' in self.topic_info.columns:
            plt.figure(figsize=(12, 6))
            valid_topics = self.topic_info[self.topic_info['Topic'] != -1]
            plt.bar(range(len(valid_topics)), valid_topics['Count'])
            plt.title('主题大小分布（改进版）')
            plt.xlabel('主题ID')
            plt.ylabel('文档数量')
            plt.xticks(range(len(valid_topics)), valid_topics['Topic'])
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'topic_size_distribution_high_quality.png'), dpi=300)
            plt.close()
        
        # 2. 主题覆盖度饼图
        total_docs = len(self.topic_docs)
        outlier_docs = len(self.topic_docs[self.topic_docs['topic'] == -1])
        covered_docs = total_docs - outlier_docs
        
        plt.figure(figsize=(8, 8))
        labels = ['有效主题文档', '离群文档']
        sizes = [covered_docs, outlier_docs]
        colors = ['#66b3ff', '#ff9999']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('主题覆盖度分布（改进版）')
        plt.axis('equal')
        plt.savefig(os.path.join(output_dir, 'topic_coverage_high_quality.png'), dpi=300)
        plt.close()
        
        # 3. 主题一致性分数分布
        coherence_scores, valid_topics = self.calculate_topic_coherence()
        if coherence_scores and len(coherence_scores) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(coherence_scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('主题一致性分数分布（改进版）')
            plt.xlabel('一致性分数')
            plt.ylabel('主题数量')
            plt.axvline(np.mean(coherence_scores), color='red', linestyle='--', 
                       label=f'平均值: {np.mean(coherence_scores):.3f}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'topic_coherence_distribution_high_quality.png'), dpi=300)
            plt.close()
        else:
            print("⚠️  主题一致性分数为空，跳过一致性分布图生成")
        
        # 4. 主题柱状图（类似BERTopic的barchart）
        if 'Count' in self.topic_info.columns:
            valid_topics = self.topic_info[self.topic_info['Topic'] != -1].head(10)
            plt.figure(figsize=(15, 8))
            bars = plt.bar(range(len(valid_topics)), valid_topics['Count'], 
                          color=sns.color_palette("husl", len(valid_topics)))
            plt.title('主题文档数量分布（前10个主题）', fontsize=16, fontweight='bold')
            plt.xlabel('主题编号', fontsize=12)
            plt.ylabel('文档数量', fontsize=12)
            plt.xticks(range(len(valid_topics)), [f'主题{t}' for t in valid_topics['Topic']], rotation=45)
            
            # 添加数值标签
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{int(height):,}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'topic_barchart_high_quality.png'), dpi=300)
            plt.close()
        
        # 5. 主题关键词热力图
        valid_topics = self.topic_info[self.topic_info['Topic'] != -1].head(8)
        topic_words_data = []
        for _, row in valid_topics.iterrows():
            topic_id = row['Topic']
            topic_words = self.topic_model.get_topic(topic_id)[:8]
            for word, score in topic_words:
                topic_words_data.append({
                    'Topic': f'主题{topic_id}',
                    'Word': word,
                    'Score': score
                })
        
        if topic_words_data:
            import pandas as pd
            words_df = pd.DataFrame(topic_words_data)
            pivot_df = words_df.pivot(index='Topic', columns='Word', values='Score').fillna(0)
            
            plt.figure(figsize=(16, 8))
            sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='YlOrRd', cbar_kws={'label': 'TF-IDF Score'})
            plt.title('主题关键词热力图（前8个主题）', fontsize=16, fontweight='bold')
            plt.xlabel('关键词', fontsize=12)
            plt.ylabel('主题', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'topic_keywords_heatmap_high_quality.png'), dpi=300)
            plt.close()
        
        # 6. 文档分布散点图（类似BERTopic的documents）
        try:
            # 使用UMAP降维到2D
            from umap import UMAP
            umap_reducer = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=42)
            reduced_embeddings = umap_reducer.fit_transform(self.embeddings)
            
            # 创建文档分布图
            plt.figure(figsize=(12, 8))
            valid_mask = self.topic_docs['topic'] != -1
            outlier_mask = self.topic_docs['topic'] == -1
            
            # 绘制有效主题文档
            if valid_mask.any():
                valid_embeddings = reduced_embeddings[valid_mask]
                valid_topics = self.topic_docs.loc[valid_mask, 'topic'].values
                scatter = plt.scatter(valid_embeddings[:, 0], valid_embeddings[:, 1], 
                                    c=valid_topics, cmap='tab20', alpha=0.6, s=20)
            
            # 绘制离群文档
            if outlier_mask.any():
                outlier_embeddings = reduced_embeddings[outlier_mask]
                plt.scatter(outlier_embeddings[:, 0], outlier_embeddings[:, 1], 
                           c='gray', alpha=0.3, s=10, label='离群文档')
            
            plt.title('文档分布图（UMAP降维）', fontsize=16, fontweight='bold')
            plt.xlabel('UMAP维度1', fontsize=12)
            plt.ylabel('UMAP维度2', fontsize=12)
            plt.colorbar(scatter, label='主题编号')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'document_distribution_umap_high_quality.png'), dpi=300)
            plt.close()
        except Exception as e:
            print(f"文档分布图生成失败: {e}")
        
        # 7. 主题概率分布直方图
        if 'topic_probability' in self.topic_docs.columns:
            plt.figure(figsize=(12, 6))
            plt.hist(self.topic_docs['topic_probability'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('主题概率分布', fontsize=16, fontweight='bold')
            plt.xlabel('主题概率', fontsize=12)
            plt.ylabel('文档数量', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'topic_probability_distribution_high_quality.png'), dpi=300)
            plt.close()
        
        # 8. 主题大小vs一致性分数散点图
        if coherence_scores and len(coherence_scores) > 0 and len(valid_topics) > 0:
            valid_topic_info = self.topic_info[self.topic_info['Topic'].isin(valid_topics)]
            if len(valid_topic_info) == len(coherence_scores):
                plt.figure(figsize=(10, 6))
                plt.scatter(valid_topic_info['Count'], coherence_scores, alpha=0.7, s=100)
                plt.title('主题大小 vs 一致性分数', fontsize=16, fontweight='bold')
                plt.xlabel('主题大小（文档数）', fontsize=12)
                plt.ylabel('一致性分数', fontsize=12)
                plt.grid(True, alpha=0.3)
                
                # 添加主题标签
                for i, (topic_id, count, coherence) in enumerate(zip(valid_topic_info['Topic'], 
                                                                     valid_topic_info['Count'], coherence_scores)):
                    plt.annotate(f'主题{topic_id}', (count, coherence), 
                                xytext=(5, 5), textcoords='offset points', fontsize=8)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'topic_size_vs_coherence_high_quality.png'), dpi=300)
                plt.close()
            else:
                print(f"⚠️  主题信息与一致性分数数量不匹配，跳过散点图生成")
        else:
            print("⚠️  主题一致性分数为空，跳过散点图生成")
        
        # 统计实际生成的文件数量
        import glob
        png_files = glob.glob(os.path.join(output_dir, '*.png'))
        print(f"✓ 实际生成了 {len(png_files)} 种可视化图表，保存在: {output_dir}")

def main():
    """主函数"""
    print("=== 微博数据BERTopic评估指标计算（改进版结果） ===")
    
    # 定义jieba_tokenizer函数（用于加载模型）
    def jieba_tokenizer(text):
        """改进的jieba分词器"""
        import jieba
        stopwords = set()
        try:
            with open('分词/stopwords.txt', 'r', encoding='utf-8') as f:
                stopwords = set([line.strip() for line in f])
        except:
            stopwords = set()
        words = [w for w in jieba.lcut(text) if w not in stopwords and w.strip()]
        return words if words else ['文本']
    
    # 检查5_2改进版结果文件是否存在
    required_files = [
        "data/high_quality_results/topic_modeling_results_high_quality.csv",
        "data/high_quality_results/topic_info_high_quality.csv",
        "embedding/emb.npy",
        "embedding/original_texts.txt"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ 缺少必要文件: {file_path}")
            print("请先运行 5_2_improve_bertopic_quality.py 生成改进版结果")
            return
    
    # 加载数据
    print("加载改进版数据...")
    
    # 加载原始文本
    with open("embedding/original_texts.txt", 'r', encoding='utf-8') as f:
        docs = [line.strip() for line in f.readlines()]
    
    # 加载嵌入向量
    embeddings = np.load("embedding/emb.npy")
    
    # 加载改进版主题文档信息
    topic_docs = pd.read_csv("data/high_quality_results/topic_modeling_results_high_quality.csv", encoding='utf-8')
    
    # 加载改进版主题信息
    topic_info = pd.read_csv("data/high_quality_results/topic_info_high_quality.csv", encoding='utf-8')
    
    # 确保嵌入向量和主题文档数量匹配
    if len(topic_docs) < len(embeddings):
        print(f"⚠️  主题文档数量({len(topic_docs):,})少于嵌入向量数量({len(embeddings):,})")
        print(f"   只使用前{len(topic_docs):,}个嵌入向量进行评估")
        embeddings = embeddings[:len(topic_docs)]
    
    print(f"✓ 加载了 {len(docs):,} 个文档")
    print(f"✓ 加载了 {embeddings.shape[0]:,} 个嵌入向量")
    print(f"✓ 加载了 {len(topic_docs):,} 个主题文档结果")
    print(f"✓ 发现 {len(topic_info)} 个主题")
    
    # 创建一个简化的模型对象用于评估
    print("创建评估用的模型对象...")
    
    class SimpleTopicModel:
        """简化的主题模型对象，用于评估"""
        def __init__(self, topic_info):
            self.topic_info = topic_info
            self._topics = {}
            
            # 从topic_info创建主题字典
            for _, row in topic_info.iterrows():
                topic_id = row['Topic']
                if topic_id != -1:
                    # 解析关键词（从Representation列）
                    representation = row['Representation']
                    if isinstance(representation, str):
                        # 解析列表格式的字符串
                        try:
                            # 移除方括号和引号，分割关键词
                            words_str = representation.strip('[]').replace("'", "").replace('"', '')
                            words = [word.strip() for word in words_str.split(',') if word.strip() and word.strip() != '']
                            # 创建(word, score)格式
                            self._topics[topic_id] = [(word, 1.0) for word in words[:10]]
                        except:
                            # 如果解析失败，使用Name列
                            name = row['Name']
                            if isinstance(name, str):
                                self._topics[topic_id] = [(name, 1.0)]
                            else:
                                self._topics[topic_id] = [('unknown', 1.0)]
                    else:
                        self._topics[topic_id] = [('unknown', 1.0)]
        
        def get_topics(self):
            return self._topics
        
        def get_topic(self, topic_id):
            return self._topics.get(topic_id, [])
        
        def get_topic_info(self):
            return self.topic_info
    
    # 创建简化的模型对象
    topic_model = SimpleTopicModel(topic_info)
    print(f"✓ 成功创建评估模型对象")
    print(f"✓ 模型主题数量: {len(topic_model.get_topics())}")
    
    # 创建评估器
    evaluator = TopicModelEvaluator(topic_model, embeddings, docs, topic_docs)
    
    # 生成评估报告（保存到改进版结果目录）
    output_dir = "data/high_quality_results"
    evaluation_results = evaluator.generate_evaluation_report(output_dir)
    
    print("\n✅ 改进版评估完成！")
    print(f"结果文件已保存到 {output_dir}/ 目录")

if __name__ == "__main__":
    # 设置日志文件
    log_file = "data/log/06_evaluation_metrics_high_quality_log.txt"
    
    # 确保日志目录存在
    os.makedirs('data/log', exist_ok=True)
    
    # 设置日志记录
    logger = Logger(log_file)
    sys.stdout = logger
    
    try:
        main()
    finally:
        # 恢复标准输出并关闭日志文件
        sys.stdout = sys.__stdout__
        logger.close()
        print(f"改进版模型评估日志已保存到: {log_file}") 