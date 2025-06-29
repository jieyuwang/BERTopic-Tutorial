#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
微博主题建模主脚本 - 使用BERTopic进行主题发现和分析
"""

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import jieba
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import os
import warnings
import time
import psutil
import traceback
warnings.filterwarnings('ignore')

# 设置并行处理
os.environ['OMP_NUM_THREADS'] = '4'  # 设置OpenMP线程数
os.environ['MKL_NUM_THREADS'] = '4'  # 设置MKL线程数
os.environ['NUMEXPR_NUM_THREADS'] = '4'  # 设置NumExpr线程数

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

def jieba_tokenizer(text):
    """jieba分词器"""
    return jieba.lcut(text)

def load_data():
    """加载数据"""
    print("=== 加载数据 ===")
    
    # 加载嵌入向量
    try:
        embeddings = np.load('embedding/emb.npy')
        print(f"成功加载嵌入向量，形状: {embeddings.shape}")
    except Exception as e:
        print(f"加载嵌入向量失败: {e}")
        return None, None
    
    # 优先加载原始文本（用于BERTopic训练）
    texts = None
    try:
        with open('embedding/original_texts.txt', 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"成功加载原始文本数据，数量: {len(texts):,}")
        print("使用原始文本进行BERTopic训练（推荐）")
    except Exception as e:
        print(f"原始文本加载失败: {e}")
        print("尝试加载分词后的文本...")
        
        # 备用：加载分词后的文本
        try:
            with open('embedding/embedding_texts.txt', 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            print(f"成功加载分词后文本数据，数量: {len(texts):,}")
            print("⚠️  使用分词后文本，可能影响BERTopic效果")
        except Exception as e2:
            print(f"加载文本数据失败: {e2}")
            return None, None
    
    if len(texts) != embeddings.shape[0]:
        print(f"警告：文本数量({len(texts)})与嵌入向量数量({embeddings.shape[0]})不匹配！")
        # 取较小的数量
        min_len = min(len(texts), embeddings.shape[0])
        texts = texts[:min_len]
        embeddings = embeddings[:min_len]
        print(f"已调整为: {min_len} 个样本")
    
    return embeddings, texts

def create_bertopic_model():
    """创建BERTopic模型"""
    print("\n=== 创建BERTopic模型 ===")
    
    # 自定义中文分词器
    vectorizer = CountVectorizer(
        analyzer='word',
        tokenizer=jieba_tokenizer,
        max_features=2000,  # 减少词汇表大小以加速
        min_df=3,  # 降低最小文档频率
        max_df=0.9,
        stop_words=None,
        ngram_range=(1, 1)  # 只使用1-gram以加速
    )
    
    # 配置BERTopic参数 - 优化版本（加速）
    topic_model = BERTopic(
        # 聚类参数
        min_topic_size=50,   # 降低最小主题大小以加速
        nr_topics=20,        # 限制主题数量为20个
        
        # 主题表示参数
        top_n_words=20,      # 适中的关键词数量
        
        # 中文分词器
        vectorizer_model=vectorizer,
        
        # 其他参数
        verbose=True,
        calculate_probabilities=True
    )
    
    print("BERTopic模型配置完成")
    print("使用优化的参数配置以加速训练")
    print("主题数量限制为20个，改善可视化效果")
    return topic_model

def train_model(topic_model, embeddings, texts):
    """训练模型"""
    print("\n=== 开始训练模型 ===")
    print(f"数据规模: {len(texts):,} 个文档")
    print(f"嵌入向量维度: {embeddings.shape[1]}")
    
    # 添加时间监控
    start_time = time.time()
    
    # 检查内存使用
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"初始内存使用: {initial_memory:.1f} MB")
    
    # 训练模型 - 优化版本
    try:
        print("开始训练BERTopic模型...")
        print("注意：训练过程可能需要几分钟，请耐心等待...")
        print("训练阶段:")
        print("1. 降维 (UMAP)")
        print("2. 聚类 (HDBSCAN)")
        print("3. 主题表示 (TF-IDF)")
        print("4. 主题概率计算")
        print("注意：主题数量已限制为20个，以改善可视化效果")
        print("-" * 50)
        
        # 分阶段训练，添加进度监控
        stage_start = time.time()
        
        # 阶段1: 降维
        print("阶段1: 开始降维 (UMAP)...")
        stage1_start = time.time()
        
        # 使用较小的数据样本进行快速测试
        if len(texts) > 20000:
            print(f"数据量较大({len(texts):,})，先使用前20,000个样本进行快速测试...")
            test_texts = texts[:20000]
            test_embeddings = embeddings[:20000]
        else:
            test_texts = texts
            test_embeddings = embeddings
        
        print(f"使用 {len(test_texts):,} 个样本进行训练")
        
        # 训练模型
        topics, probs = topic_model.fit_transform(test_texts, test_embeddings)
        
        stage1_time = time.time() - stage1_start
        current_memory = process.memory_info().rss / 1024 / 1024
        print(f"✓ 阶段1完成，用时: {stage1_time:.1f}秒")
        print(f"✓ 当前内存使用: {current_memory:.1f} MB")
        
        training_time = time.time() - start_time
        print(f"✓ 训练完成！")
        print(f"✓ 总训练用时: {training_time/60:.1f} 分钟")
        print(f"✓ 发现主题数量: {len(topic_model.get_topics())}")
        
        # 如果使用了测试样本，询问是否用完整数据重新训练
        if len(test_texts) < len(texts):
            print(f"\n快速测试完成！")
            print(f"测试样本: {len(test_texts):,} 个")
            print(f"完整数据: {len(texts):,} 个")
            print(f"发现主题: {len(topic_model.get_topics())} 个")
            print(f"\n建议：")
            print(f"1. 如果测试结果满意，可以继续使用当前模型")
            print(f"2. 如果需要更全面的结果，可以运行完整数据训练")
            print(f"3. 完整训练预计需要 {training_time * (len(texts)/len(test_texts)) / 60:.1f} 分钟")
        
        return topics, probs
        
    except Exception as e:
        print(f"✗ 训练失败: {e}")
        print("错误详情:")
        traceback.print_exc()
        print("\n尝试使用备用方法...")
        
        # 备用方法：只使用文本，不使用预计算的嵌入向量
        try:
            print("使用备用方法：仅使用文本进行训练...")
            print("注意：这将重新计算嵌入向量，可能需要更长时间")
            
            # 使用较小的样本
            if len(texts) > 10000:
                print(f"使用前10,000个样本进行测试...")
                test_texts = texts[:10000]
            else:
                test_texts = texts
            
            topics, probs = topic_model.fit_transform(test_texts)
            
            training_time = time.time() - start_time
            print(f"✓ 备用方法训练完成！")
            print(f"✓ 训练用时: {training_time/60:.1f} 分钟")
            print(f"✓ 发现主题数量: {len(topic_model.get_topics())}")
            
            return topics, probs
            
        except Exception as e2:
            print(f"✗ 备用方法也失败: {e2}")
            print("错误详情:")
            traceback.print_exc()
            return None, None

def analyze_topics(topic_model, topics, texts):
    """分析主题"""
    print("\n=== 主题分析 ===")
    
    # 获取主题信息
    topic_info = topic_model.get_topic_info()
    print(f"主题统计信息:")
    print(topic_info.head(10))
    
    # 统计主题分布
    topic_counts = pd.Series(topics).value_counts().sort_index()
    print(f"\n主题分布:")
    for topic_id, count in topic_counts.head(10).items():
        if topic_id != -1:  # 排除异常值主题
            topic_words = topic_model.get_topic(topic_id)[:5]
            words_str = ', '.join([word for word, _ in topic_words])
            print(f"主题 {topic_id}: {count:,} 个文档 - {words_str}")
        else:
            print(f"异常值主题: {count:,} 个文档")
    
    return topic_info, topic_counts

def save_results(topic_model, topics, probs, texts, topic_info):
    """保存结果"""
    print("\n=== 保存结果 ===")
    
    # 处理概率数组
    if probs is not None:
        if probs.ndim == 1:
            # 如果是一维数组，直接使用
            topic_probabilities = probs
        else:
            # 如果是二维数组，取最大值
            topic_probabilities = np.max(probs, axis=1)
    else:
        # 如果没有概率信息，设为1.0
        topic_probabilities = [1.0] * len(texts)
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'text': texts,
        'topic': topics,
        'topic_probability': topic_probabilities
    })
    
    # 添加主题标签
    topic_labels = topic_model.get_topic_info()
    topic_label_dict = dict(zip(topic_labels['Topic'], topic_labels['Name']))
    results_df['topic_label'] = results_df['topic'].map(topic_label_dict)
    
    # 保存结果到文件
    results_file = "results/standard_results/topic_modeling_results.csv"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    results_df.to_csv(results_file, index=False, encoding='utf-8')
    print(f"✓ 主题建模结果已保存到: {results_file}")
    
    # 保存主题信息
    topic_info_file = "results/standard_results/topic_info.csv"
    topic_info.to_csv(topic_info_file, index=False, encoding='utf-8')
    print(f"✓ 主题信息已保存到: {topic_info_file}")
    
    # 保存模型
    model_file = "results/standard_results/bertopic_model"
    topic_model.save(model_file)
    print(f"✓ BERTopic模型已保存到: {model_file}")
    
    return results_df

def create_visualizations(topic_model, topic_info):
    """创建可视化图表"""
    print("\n=== 创建可视化图表 ===")
    
    # 创建输出目录
    os.makedirs('results', exist_ok=True)
    
    # 1. 主题数量分布图
    plt.figure(figsize=(12, 6))
    topic_counts = topic_info['Count'].values
    topic_names = topic_info['Name'].values
    
    plt.bar(range(len(topic_counts)), topic_counts)
    plt.title('主题文档数量分布')
    plt.xlabel('主题')
    plt.ylabel('文档数量')
    plt.xticks(range(len(topic_names)), topic_names, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/topic_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("主题分布图已保存到: results/topic_distribution.png")
    
    # 2. 主题层次聚类图
    try:
        fig = topic_model.visualize_hierarchy()
        fig.write_html('results/topic_hierarchy.html')
        print("主题层次图已保存到: results/topic_hierarchy.html")
    except Exception as e:
        print(f"主题层次图生成失败: {e}")
    
    # 3. 主题相似度图
    try:
        fig = topic_model.visualize_heatmap()
        fig.write_html('results/topic_similarity.html')
        print("主题相似度图已保存到: results/topic_similarity.html")
    except Exception as e:
        print(f"主题相似度图生成失败: {e}")
    
    # 4. 主题词云图
    try:
        fig = topic_model.visualize_topics()
        fig.write_html('results/topic_visualization.html')
        print("主题可视化图已保存到: results/topic_visualization.html")
    except Exception as e:
        print(f"主题可视化图生成失败: {e}")

def generate_report(topic_model, topic_info, results_df):
    """生成分析报告"""
    print("\n=== 生成分析报告 ===")
    
    # 计算统计信息
    total_docs = len(results_df)
    outlier_docs = len(results_df[results_df['topic'] == -1])
    valid_topics = len(topic_info) - 1  # 排除异常值主题
    
    avg_prob = results_df['topic_probability'].mean()
    min_prob = results_df['topic_probability'].min()
    max_prob = results_df['topic_probability'].max()
    
    # 生成报告
    report = f"""
微博主题建模分析报告
{'='*50}

数据概览:
- 总文档数: {total_docs:,}
- 有效主题数: {valid_topics}
- 异常值文档数: {outlier_docs:,} ({outlier_docs/total_docs*100:.1f}%)
- 平均主题概率: {avg_prob:.3f}
- 主题概率范围: {min_prob:.3f} - {max_prob:.3f}

主题详情:
"""
    
    # 添加每个主题的详细信息
    for _, row in topic_info.head(10).iterrows():
        topic_id = row['Topic']
        if topic_id != -1:  # 排除异常值主题
            count = row['Count']
            name = row['Name']
            percentage = count / total_docs * 100
            
            # 获取主题关键词
            topic_words = topic_model.get_topic(topic_id)[:10]
            words_str = ', '.join([word for word, _ in topic_words])
            
            report += f"""
主题 {topic_id}: {name}
- 文档数: {count:,} ({percentage:.1f}%)
- 关键词: {words_str}
"""
    
    # 保存报告
    with open('results/5_1_main_results/topic_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("分析报告已保存到: results/topic_analysis_report.txt")
    print(report)

def main():
    """主函数"""
    print("=" * 60)
    print("微博主题建模工具")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 加载数据
    embeddings, texts = load_data()
    if embeddings is None or texts is None:
        return
    
    # 创建模型
    topic_model = create_bertopic_model()
    
    # 训练模型
    topics, probs = train_model(topic_model, embeddings, texts)
    if topics is None:
        print("模型训练失败，程序退出")
        return
    
    # 分析主题
    topic_info, topic_counts = analyze_topics(topic_model, topics, texts)
    
    # 保存结果
    results_df = save_results(topic_model, topics, probs, texts, topic_info)
    
    # 创建可视化
    create_visualizations(topic_model, topic_info)
    
    # 生成报告
    generate_report(topic_model, topic_info, results_df)
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("主题建模完成！")
    print("=" * 60)
    
    print(f"\n输出文件:")
    print(f"1. results/standard_results/topic_modeling_results.csv - 标准版本主题建模结果")
    print(f"2. results/standard_results/topic_info.csv - 标准版本主题信息")
    print(f"3. results/standard_results/bertopic_model - 标准版本模型")
    print(f"4. data/log/05_1_main_weibo_log.txt - 标准版本日志")
    print(f"\n下一步: 运行 6_evaluation_metrics.py 进行模型评估")

if __name__ == "__main__":
    # 设置日志文件
    log_file = "data/log/05_1_main_weibo_log.txt"
    
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
        print(f"主题建模日志已保存到: {log_file}") 