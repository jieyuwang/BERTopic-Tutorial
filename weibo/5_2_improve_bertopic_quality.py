#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的BERTopic主题建模 - 提高主题质量
"""

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import jieba
import re
import os
import sys
from datetime import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志功能
def setup_logging():
    """设置日志功能"""
    # 创建log目录
    log_dir = "data/log"
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/05_2_improve_bertopic_quality_log.txt"
    
    # 创建日志文件并重定向输出
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w', encoding='utf-8')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    # 重定向标准输出到日志文件
    sys.stdout = Logger(log_file)
    
    print(f"日志文件: {log_file}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    return log_file

# 设置并行处理
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # 移除URL
    text = re.sub(r'http[s]?://\S+', '', text)
    # 移除邮箱
    text = re.sub(r'\S+@\S+', '', text)
    # 移除特殊符号，保留中英文、数字、常用标点
    text = re.sub(r'[^\u4e00-\u9fff0-9a-zA-Z，。！？；：、…""''（）《》【】\s]', '', text)
    # 移除多余空白
    text = re.sub(r'\s+', '', text)
    return text

def jieba_tokenizer(text):
    """改进的jieba分词器 - 更强力过滤"""
    stopwords = set()
    try:
        with open('分词/stopwords.txt', 'r', encoding='utf-8') as f:
            stopwords = set([line.strip() for line in f])
    except:
        stopwords = set()
    words = [w for w in jieba.lcut(text) if w not in stopwords and w.strip()]
    return words if words else ['文本']

def load_and_preprocess_data():
    """加载和预处理数据"""
    print("=== 加载和预处理数据 ===")
    
    # 加载嵌入向量
    try:
        embeddings = np.load('embedding/emb.npy')
        print(f"嵌入向量形状: {embeddings.shape}")
    except Exception as e:
        print(f"加载嵌入向量失败: {e}")
        return None, None
    
    # 加载原始文本
    try:
        with open('embedding/original_texts.txt', 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"原始文本数量: {len(texts):,}")
    except Exception as e:
        print(f"加载原始文本失败: {e}")
        return None, None
    
    # 预处理文本
    print("预处理文本...")
    processed_texts = []
    valid_indices = []
    
    for i, text in enumerate(texts):
        cleaned_text = clean_text(text)
        if len(cleaned_text) > 10:  # 只保留长度大于10的文本
            processed_texts.append(cleaned_text)
            valid_indices.append(i)
    
    # 过滤嵌入向量
    filtered_embeddings = embeddings[valid_indices]
    
    print(f"预处理后文本数量: {len(processed_texts):,}")
    print(f"过滤后嵌入向量形状: {filtered_embeddings.shape}")
    
    # 显示预处理示例
    print("\n预处理示例:")
    for i in range(min(3, len(processed_texts))):
        print(f"原文{i+1}: {texts[valid_indices[i]][:50]}...")
        print(f"处理后: {processed_texts[i][:50]}...")
        print()
    
    return filtered_embeddings, processed_texts

def create_high_quality_model():
    """创建高质量的BERTopic模型"""
    print("\n=== 创建高质量BERTopic模型 ===")
    
    # 高质量分词器配置
    vectorizer = CountVectorizer(
        analyzer='word',
        tokenizer=jieba_tokenizer,
        max_features=5000,
        min_df=2,
        max_df=0.9,
        stop_words=None,
        ngram_range=(1, 1)
    )
    
    # 配置BERTopic参数 - 高质量版本
    topic_model = BERTopic(
        # 聚类参数
        min_topic_size=150,  # 提高最小主题大小以获得更稳定的主题
        nr_topics=20,        # 限制主题数量为20个
        
        # 主题表示参数
        top_n_words=20,      # 增加关键词数量，便于分析
        
        # 中文分词器
        vectorizer_model=vectorizer,
        
        # 其他参数
        verbose=True,
        calculate_probabilities=True
    )
    
    print("✓ 改进版BERTopic模型配置完成")
    print("✓ 使用强力文本清洗和过滤策略")
    print("✓ 限制主题数量为20个，改善可视化效果")
    return topic_model

def train_high_quality_model(topic_model, embeddings, texts, sample_size=None):
    """训练改进版模型"""
    print("\n=== 训练改进版模型 ===")
    
    if sample_size and sample_size < len(texts):
        print(f"使用前 {sample_size:,} 个样本进行训练")
        train_texts = texts[:sample_size]
        train_embeddings = embeddings[:sample_size]
    else:
        print(f"使用全部 {len(texts):,} 个样本进行训练")
        train_texts = texts
        train_embeddings = embeddings
    
    print(f"训练数据规模: {len(train_texts):,} 个文档")
    print(f"嵌入向量维度: {train_embeddings.shape[1]}")
    
    start_time = time.time()
    
    try:
        print("开始训练...")
        topics, probs = topic_model.fit_transform(train_texts, train_embeddings)
        
        training_time = time.time() - start_time
        print(f"✓ 训练完成！")
        print(f"✓ 训练用时: {training_time/60:.1f} 分钟")
        print(f"✓ 发现主题数量: {len(topic_model.get_topics())}")
        
        return topics, probs
        
    except Exception as e:
        print(f"✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def analyze_high_quality_results(topic_model, topics, texts):
    """分析改进版结果"""
    print("\n=== 改进版结果分析 ===")
    
    # 获取主题信息
    topic_info = topic_model.get_topic_info()
    print(f"主题统计信息:")
    print(topic_info.head(10))
    
    # 统计主题分布
    topic_counts = pd.Series(topics).value_counts().sort_index()
    print(f"\n主题分布 (前15个):")
    
    for topic_id, count in topic_counts.head(15).items():
        if topic_id != -1:
            topic_words = topic_model.get_topic(topic_id)[:10]
            words_str = ', '.join([word for word, _ in topic_words])
            print(f"主题 {topic_id}: {count:,} 个文档 - {words_str}")
        else:
            print(f"异常值主题: {count:,} 个文档")
    
    # 统计异常值
    outlier_count = len([t for t in topics if t == -1])
    print(f"\n异常值统计:")
    print(f"  异常值数量: {outlier_count:,}")
    print(f"  异常值比例: {outlier_count/len(topics)*100:.1f}%")
    
    return topic_info, topic_counts

def save_results(topic_model, results_df, topic_info, texts):
    """保存改进版结果"""
    print("\n=== 保存改进版结果 ===")
    
    # 创建高质量结果目录
    output_dir = "results/high_quality_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ 创建输出目录: {output_dir}")
    
    # 保存主题建模结果
    results_file = f"{output_dir}/topic_modeling_results_high_quality.csv"
    results_df.to_csv(results_file, index=False, encoding='utf-8')
    print(f"✓ 主题建模结果已保存到: {results_file}")
    
    # 保存主题信息
    topic_info_file = f"{output_dir}/topic_info_high_quality.csv"
    topic_info.to_csv(topic_info_file, index=False, encoding='utf-8')
    print(f"✓ 主题信息已保存到: {topic_info_file}")
    
    # 保存模型
    model_file = f"{output_dir}/bertopic_model_high_quality"
    topic_model.save(model_file)
    print(f"✓ BERTopic模型已保存到: {model_file}")
    
    return output_dir

def create_visualizations(topic_model, topic_info, results_df, output_dir):
    """创建可视化图表"""
    print("\n=== 创建可视化图表 ===")
    
    # 创建可视化目录
    viz_dir = f"{output_dir}/visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. 主题分布柱状图
    print("创建主题分布图...")
    valid_topics = topic_info[topic_info['Topic'] != -1].head(15)
    
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(valid_topics)), valid_topics['Count'], 
                   color=sns.color_palette("husl", len(valid_topics)))
    plt.title('主题文档数量分布 (前15个主题)', fontsize=16, fontweight='bold')
    plt.xlabel('主题编号', fontsize=12)
    plt.ylabel('文档数量', fontsize=12)
    plt.xticks(range(len(valid_topics)), [f'主题{t}' for t in valid_topics['Topic']], rotation=45)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height):,}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/topic_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 主题分布图已保存到: {viz_dir}/topic_distribution.png")
    
    # 2. 主题关键词热力图
    print("创建主题关键词热力图...")
    topic_words_data = []
    for _, row in valid_topics.head(10).iterrows():
        topic_id = row['Topic']
        topic_words = topic_model.get_topic(topic_id)[:10]
        for word, score in topic_words:
            topic_words_data.append({
                'Topic': f'主题{topic_id}',
                'Word': word,
                'Score': score
            })
    
    if topic_words_data:
        words_df = pd.DataFrame(topic_words_data)
        pivot_df = words_df.pivot(index='Topic', columns='Word', values='Score').fillna(0)
        
        plt.figure(figsize=(20, 10))
        sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='YlOrRd', cbar_kws={'label': 'TF-IDF Score'})
        plt.title('主题关键词热力图 (前10个主题)', fontsize=16, fontweight='bold')
        plt.xlabel('关键词', fontsize=12)
        plt.ylabel('主题', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{viz_dir}/topic_keywords_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 主题关键词热力图已保存到: {viz_dir}/topic_keywords_heatmap.png")
    
    # 3. 文档分布饼图
    print("创建文档分布饼图...")
    total_docs = len(results_df)
    outlier_docs = len(results_df[results_df['topic'] == -1])
    valid_docs = total_docs - outlier_docs
    
    plt.figure(figsize=(12, 5))
    
    # 文档分布
    plt.subplot(1, 2, 1)
    labels = ['有效文档', '异常值文档']
    sizes = [valid_docs, outlier_docs]
    colors = ['lightgreen', 'lightcoral']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('文档分布', fontsize=14, fontweight='bold')
    
    # 主题大小分布
    plt.subplot(1, 2, 2)
    valid_topics = topic_info[topic_info['Topic'] != -1]
    topic_sizes = valid_topics['Count'].values
    
    # 按大小分类
    small_topics = len(topic_sizes[topic_sizes < 100])
    medium_topics = len(topic_sizes[(topic_sizes >= 100) & (topic_sizes < 500)])
    large_topics = len(topic_sizes[topic_sizes >= 500])
    
    labels = ['小主题 (<100)', '中主题 (100-500)', '大主题 (≥500)']
    sizes = [small_topics, medium_topics, large_topics]
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('主题大小分布', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/document_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 文档分布饼图已保存到: {viz_dir}/document_distribution.png")
    
    # 4. 主题概率分布直方图
    print("创建主题概率分布图...")
    plt.figure(figsize=(12, 6))
    plt.hist(results_df['topic_probability'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('主题概率分布', fontsize=16, fontweight='bold')
    plt.xlabel('主题概率', fontsize=12)
    plt.ylabel('文档数量', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/topic_probability_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 主题概率分布图已保存到: {viz_dir}/topic_probability_distribution.png")
    
    return viz_dir

def generate_high_quality_report(topic_model, topic_info, results_df, viz_dir):
    """生成改进版分析报告"""
    print("\n=== 生成改进版分析报告 ===")
    
    # 计算统计信息
    total_docs = len(results_df)
    outlier_docs = len(results_df[results_df['topic'] == -1])
    valid_topics = len(topic_info) - 1  # 排除异常值主题
    
    avg_prob = results_df['topic_probability'].mean()
    min_prob = results_df['topic_probability'].min()
    max_prob = results_df['topic_probability'].max()
    
    # 生成报告
    report = f"""
改进版微博主题建模分析报告
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
    for _, row in topic_info.head(15).iterrows():
        topic_id = row['Topic']
        if topic_id != -1:  # 排除异常值主题
            count = row['Count']
            name = row['Name']
            percentage = count / total_docs * 100
            
            # 获取主题关键词
            topic_words = topic_model.get_topic(topic_id)[:15]
            words_str = ', '.join([word for word, _ in topic_words])
            
            report += f"""
主题 {topic_id}: {name}
- 文档数: {count:,} ({percentage:.1f}%)
- 关键词: {words_str}
"""
    
    report += f"""
改进说明:
- 强力文本清洗：移除所有英文单词、数字、方括号内容
- 扩展停用词：过滤更多无意义的中文词汇
- 严格分词过滤：只保留有意义的中文词汇
- 优化参数配置：提高最小文档频率，减少词汇表大小
- 增强聚类参数：提高最小主题大小，获得更稳定的主题

可视化文件:
1. {viz_dir}/topic_distribution.png - 主题分布图
2. {viz_dir}/topic_keywords_heatmap.png - 主题关键词热力图
3. {viz_dir}/document_distribution.png - 文档分布饼图
4. {viz_dir}/topic_probability_distribution.png - 主题概率分布图
"""
    
    # 保存报告
    report_file = f"{viz_dir}/improved_topic_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ 改进版分析报告已保存到: {report_file}")
    print(report)

def main():
    """主函数"""
    # 设置日志
    log_file = setup_logging()
    
    print("=" * 60)
    print("改进版BERTopic主题建模工具（强力文本清洗）")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 加载和预处理数据
    embeddings, texts = load_and_preprocess_data()
    if embeddings is None or texts is None:
        return
    
    # 创建高质量模型
    topic_model = create_high_quality_model()
    
    # 训练模型（先用小样本测试）
    print("\n先使用前15,000个样本进行测试...")
    topics, probs = train_high_quality_model(topic_model, embeddings, texts, sample_size=15000)
    if topics is None:
        print("小样本训练失败，程序退出")
        return
    
    # 分析结果
    topic_info, topic_counts = analyze_high_quality_results(topic_model, topics, texts[:15000])
    
    # 创建结果DataFrame
    if probs is not None:
        if probs.ndim == 1:
            topic_probabilities = probs
        else:
            topic_probabilities = np.max(probs, axis=1)
    else:
        topic_probabilities = [1.0] * len(topics)
    
    results_df = pd.DataFrame({
        'text': texts[:15000],
        'topic': topics,
        'topic_probability': topic_probabilities
    })
    
    # 添加主题标签
    topic_label_dict = dict(zip(topic_info['Topic'], topic_info['Name']))
    results_df['topic_label'] = results_df['topic'].map(topic_label_dict)
    
    # 保存结果
    output_dir = save_results(topic_model, results_df, topic_info, texts[:15000])
    
    # 创建可视化图表
    viz_dir = create_visualizations(topic_model, topic_info, results_df, output_dir)
    
    # 生成报告
    generate_high_quality_report(topic_model, topic_info, results_df, viz_dir)
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("改进版BERTopic主题建模完成！")
    print("=" * 60)
    
    print(f"\n输出文件:")
    print(f"1. {output_dir}/topic_modeling_results_high_quality.csv - 改进版主题建模结果")
    print(f"2. {output_dir}/topic_info_high_quality.csv - 改进版主题信息")
    print(f"3. {output_dir}/bertopic_model_high_quality - 改进版模型")
    print(f"4. {viz_dir}/improved_topic_analysis_report.txt - 改进版分析报告")
    print(f"5. {viz_dir}/topic_distribution.png - 主题分布图")
    print(f"6. {viz_dir}/topic_keywords_heatmap.png - 主题关键词热力图")
    print(f"7. {viz_dir}/document_distribution.png - 文档分布饼图")
    print(f"8. {viz_dir}/topic_probability_distribution.png - 主题概率分布图")
    
    print(f"\n日志文件: {log_file}")
    print("=" * 60)
    print("改进版BERTopic主题建模完成！")
    print("=" * 60)

if __name__ == "__main__":
    main() 