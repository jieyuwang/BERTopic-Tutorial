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
from collections import Counter

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
    log_file = f"{log_dir}/05_3_improve_bertopic_quality_log.txt"
    
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
    # 直接返回原文，不做任何处理
    return text

def load_and_preprocess_data():
    """加载和预处理数据 - 使用已分词的文本，并清洗只保留中文词汇"""
    print("=== 加载和预处理数据 ===")
    
    # 加载嵌入向量
    try:
        embeddings = np.load('embedding/emb.npy')
        print(f"嵌入向量形状: {embeddings.shape}")
    except Exception as e:
        print(f"加载嵌入向量失败: {e}")
        return None, None
    
    # 加载已分词的文本（使用正确的数据源）
    try:
        with open('data/切词.txt', 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"已分词文本数量: {len(texts):,}")
    except Exception as e:
        print(f"加载已分词文本失败: {e}")
        return None, None
    
    # 清洗文本：只保留中文词汇
    print("清洗文本：只保留中文词汇...")
    cleaned_texts = []
    valid_indices = []
    
    for i, text in enumerate(texts):
        # 按空格分割词汇
        words = text.split()
        # 只保留中文词汇（降低长度要求）
        chinese_words = []
        for word in words:
            word = word.strip()
            # 只保留纯中文词汇，长度>=1（降低要求）
            if len(word) >= 1 and re.match(r'^[\u4e00-\u9fff]+$', word):
                chinese_words.append(word)
        
        # 如果清洗后有中文词汇，则保留（降低要求）
        if len(chinese_words) >= 1:  # 至少1个中文词汇
            cleaned_text = ' '.join(chinese_words)
            cleaned_texts.append(cleaned_text)
            valid_indices.append(i)
    
    # 过滤嵌入向量
    filtered_embeddings = embeddings[valid_indices]
    
    print(f"清洗后文本数量: {len(cleaned_texts):,}")
    print(f"过滤后嵌入向量形状: {filtered_embeddings.shape}")
    print(f"清洗比例: {len(cleaned_texts)/len(texts)*100:.1f}%")
    
    # 统计清洗后的词汇
    all_words = []
    for text in cleaned_texts:
        all_words.extend(text.split())
    unique_words = set(all_words)
    print(f"清洗后总词汇数: {len(all_words):,}")
    print(f"清洗后唯一词汇数: {len(unique_words):,}")
    print(f"平均每文档词汇数: {len(all_words)/len(cleaned_texts):.1f}")
    
    # 显示清洗示例
    print("\n清洗示例:")
    for i in range(min(3, len(cleaned_texts))):
        print(f"原文{i+1}: {texts[valid_indices[i]][:100]}...")
        print(f"清洗后: {cleaned_texts[i][:100]}...")
        print()
    
    return filtered_embeddings, cleaned_texts

def create_high_quality_model():
    """创建高质量的BERTopic模型"""
    print("\n=== 创建高质量BERTopic模型 ===")
    
    # 使用BERTopic的默认配置，不传入自定义vectorizer
    topic_model = BERTopic(
        min_topic_size=10,  # 更小的主题最小文档数
        nr_topics=None,     # 不限制主题数
        verbose=True
    )
    print("✓ 默认配置BERTopic模型完成")
    print("✓ 使用BERTopic默认参数")
    return topic_model

def train_high_quality_model(topic_model, embeddings, texts, sample_size=None):
    """训练改进版模型"""
    print("\n=== 训练改进版模型 ===")
    
    # 只用前1000条数据训练
    print(f"使用前 1,000 个样本进行训练（调试模式）")
    train_texts = texts[:1000]
    train_embeddings = embeddings[:1000]
    
    # 过滤掉所有分词后为空的文本
    filtered = [(t, e) for t, e in zip(train_texts, train_embeddings) if t.strip()]
    if len(filtered) < len(train_texts):
        print(f"过滤掉空文本: {len(train_texts) - len(filtered)} 条")
    else:
        print("没有发现空文本")
    train_texts, train_embeddings = zip(*filtered)
    train_texts = list(train_texts)
    train_embeddings = np.array(train_embeddings)
    
    # 确认过滤后的文本都非空
    empty_count = sum(1 for t in train_texts if not t.strip())
    print(f"过滤后空文本数量: {empty_count}")
    if empty_count > 0:
        print("警告：过滤后仍有空文本！")
        for i, text in enumerate(train_texts):
            if not text.strip():
                print(f"  空文本{i}: '{text}'")
                break
    
    print(f"训练数据规模: {len(train_texts):,} 个文档")
    print(f"嵌入向量维度: {train_embeddings.shape[1]}")
    
    # 检查非空文本数量
    non_empty = [t for t in train_texts if t.strip()]
    print(f"非空文本数量: {len(non_empty)} / {len(train_texts)}")
    # 检查大批量文本能否提取词汇
    try:
        topic_model.vectorizer_model.fit(non_empty)
        vocab = topic_model.vectorizer_model.get_feature_names_out()
        print(f'大批量文本词汇数量: {len(vocab)}')
        print(f'词汇示例: {vocab[:20]}')
    except Exception as e:
        print(f'vectorizer批量fit错误: {e}')
    
    # 调试：检查CountVectorizer输出
    print("\n=== 调试CountVectorizer输出 ===")
    test_texts = train_texts[:3]
    for i, text in enumerate(test_texts):
        print(f"原文{i+1}: {text[:50]}...")
        # 测试CountVectorizer
        from sklearn.feature_extraction.text import CountVectorizer
        test_vectorizer = CountVectorizer(
            analyzer='word',
            token_pattern=r"\S+",  # 匹配任何非空白字符序列
            min_df=1,
            max_df=1.0
        )
        try:
            test_vectorizer.fit([text])
            vocab = test_vectorizer.get_feature_names_out()
            print(f"词汇数量: {len(vocab)}")
            print(f"词汇示例: {vocab[:10]}")
        except Exception as e:
            print(f"CountVectorizer错误: {e}")
        print()
    
    start_time = time.time()
    
    try:
        print("开始训练...")
        
        # 详细调试：检查训练文本内容
        print("\n=== 详细调试训练文本 ===")
        print(f"训练文本数量: {len(train_texts)}")
        print("前5条训练文本:")
        for i, text in enumerate(train_texts[:5]):
            print(f"  文本{i+1}: '{text[:100]}...'")
            print(f"  长度: {len(text)}, 是否为空: {not text.strip()}")
        
        # 检查是否有特殊字符
        special_chars = []
        for text in train_texts:
            if any(char in text for char in ['\n', '\t', '\r', '\x00']):
                special_chars.append(text[:50])
        if special_chars:
            print(f"发现包含特殊字符的文本: {special_chars[:3]}")
        
        # 手动调试BERTopic聚类过程
        print("\n=== 手动调试BERTopic聚类过程 ===")
        try:
            # 1. 降维
            print("1. 降维...")
            reduced_embeddings = topic_model.umap_model.fit_transform(train_embeddings)
            print(f"   降维后形状: {reduced_embeddings.shape}")
            
            # 2. 聚类
            print("2. 聚类...")
            clusters = topic_model.hdbscan_model.fit_predict(reduced_embeddings)
            unique_clusters = set(clusters)
            print(f"   聚类结果: {unique_clusters}")
            print(f"   聚类数量: {len(unique_clusters)}")
            
            # 3. 统计每个聚类的文档数量
            cluster_counts = Counter(clusters)
            print("   每个聚类的文档数量:")
            for cluster_id, count in cluster_counts.most_common():
                print(f"     聚类{cluster_id}: {count} 个文档")
            
            # 4. 检查是否有空聚类
            empty_clusters = [cid for cid, count in cluster_counts.items() if count == 0]
            if empty_clusters:
                print(f"   警告：存在空聚类: {empty_clusters}")
            
            # 5. 模拟_c_tf_idf过程：合并每个聚类的文档
            print("5. 模拟_c_tf_idf过程...")
            documents_per_topic = {}
            for cluster_id in unique_clusters:
                if cluster_id != -1:  # 跳过异常值聚类
                    cluster_docs = [train_texts[i] for i, c in enumerate(clusters) if c == cluster_id]
                    if cluster_docs:
                        merged_doc = " ".join(cluster_docs)
                        documents_per_topic[cluster_id] = merged_doc
                        print(f"   聚类{cluster_id}: {len(cluster_docs)}个文档，合并后长度: {len(merged_doc)}")
                        if not merged_doc.strip():
                            print(f"   警告：聚类{cluster_id}合并后为空字符串！")
            
            # 6. 测试vectorizer.fit
            print("6. 测试vectorizer.fit...")
            try:
                topic_model.vectorizer_model.fit(list(documents_per_topic.values()))
                vocab = topic_model.vectorizer_model.get_feature_names_out()
                print(f"   成功提取词汇: {len(vocab)}个")
                print(f"   词汇示例: {vocab[:10]}")
            except Exception as e:
                print(f"   vectorizer.fit失败: {e}")
                # 检查每个合并文档的内容
                for topic_id, doc in documents_per_topic.items():
                    if not doc.strip():
                        print(f"   聚类{topic_id}合并文档为空: '{doc}'")
                    elif len(doc.strip()) < 10:
                        print(f"   聚类{topic_id}合并文档过短: '{doc}'")
            
        except Exception as e:
            print(f"   手动调试失败: {e}")
        
        topics, probs = topic_model.fit_transform(train_texts, train_embeddings)
        training_time = time.time() - start_time
        print(f"✓ 训练完成！")
        print(f"✓ 训练用时: {training_time/60:.1f} 分钟")
        print(f"✓ 发现主题数量: {len(topic_model.get_topics())}")

        # 训练后，统计每个主题下的文档数量
        topic_counts = Counter(topics)
        print("每个主题下的文档数量（前10个主题）：")
        for topic_id, count in topic_counts.most_common(10):
            print(f"  主题{topic_id}: {count} 个文档")
        # 检查是否有空主题
        empty_topics = [tid for tid, count in topic_counts.items() if count == 0]
        if empty_topics:
            print(f"警告：存在空主题: {empty_topics}")
        
        # 检查异常值主题
        outlier_count = topic_counts.get(-1, 0)
        print(f"异常值主题(-1)文档数量: {outlier_count}")
        if outlier_count == len(topics):
            print("警告：所有文档都被分到了异常值主题，聚类可能失败！")
        
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
    
    # 加载时间戳
    timestamps = None
    try:
        with open('data/时间.txt', 'r', encoding='utf-8') as f:
            timestamps = [line.strip() for line in f]
        if len(timestamps) != len(texts):
            print(f"警告：时间戳数量({len(timestamps)})与文本数量({len(texts)})不一致，跳过topics_over_time分析")
            timestamps = None
    except Exception as e:
        print(f"加载时间戳失败: {e}")
        timestamps = None
    
    # 主题随时间分析
    if timestamps:
        try:
            topics_over_time = topic_model.topics_over_time(texts, timestamps, global_tuning=False, evolution_tuning=False)
            topic_model.visualize_topics_over_time(topics_over_time)
        except Exception as e:
            print(f"topics_over_time分析失败: {e}")
    
    print(topic_model.visualize_barchart())
    print(topic_model.visualize_topics())
    print(topic_model.visualize_topics())

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
    """保存改进版结果，并兼容main.ipynb风格输出"""
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

    # ========== 兼容main.ipynb风格的聚类结果保存 ==========
    # 1. 保存聚类结果.csv
    csv1 = "data/聚类结果.csv"
    results_df.to_csv(csv1, index=False, encoding='utf-8')
    print(f"✓ 聚类结果已保存到: {csv1}")
    # 2. 保存聚类结果2.csv（含原文和时间）
    csv2 = "data/聚类结果2.csv"
    df2 = results_df.copy()
    # 加载原文
    try:
        with open('data/文本.txt', 'r', encoding='utf-8') as f:
            raw_texts = [line.strip() for line in f]
        if len(raw_texts) == len(df2):
            df2.insert(1, '原文', raw_texts)
    except Exception as e:
        print(f"加载原文失败: {e}")
    # 加载时间
    try:
        with open('data/时间.txt', 'r', encoding='utf-8') as f:
            times = [line.strip() for line in f]
        if len(times) == len(df2):
            df2.insert(2, '时间', times)
    except Exception as e:
        print(f"加载时间失败: {e}")
    df2.to_csv(csv2, index=False, encoding='utf-8')
    print(f"✓ 聚类结果2已保存到: {csv2}")
    # ========== END ==========

    return output_dir

def create_visualizations(topic_model, topic_info, results_df, output_dir, texts=None):
    """创建可视化图表"""
    print("\n=== 创建可视化图表 ===")
    
    # 创建可视化目录
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. 主题分布图
    print("创建主题分布图...")
    try:
        fig = topic_model.visualize_barchart()
        fig.write_image(os.path.join(viz_dir, "topic_distribution.png"))
        
        # 同时保存到data/image目录
        data_image_dir = "data/image"
        os.makedirs(data_image_dir, exist_ok=True)
        fig.write_image(os.path.join(data_image_dir, "topic_barchart_high_quality.png"))
        print("✓ 主题分布图已保存到: results/high_quality_results/visualizations/topic_distribution.png 和 data/image/topic_barchart_high_quality.png")
    except Exception as e:
        print(f"✗ 主题分布图创建失败: {e}")
    
    # 2. 主题关键词热力图
    print("创建主题关键词热力图...")
    try:
        # 获取主题词汇数据
        words_df = []
        for topic_id in topic_info['Topic'].values:
            if topic_id != -1:  # 排除异常值主题
                topic_words = topic_model.get_topic(topic_id)
                for word, score in topic_words:
                    words_df.append({
                        'Topic': f'Topic {topic_id}',
                        'Word': word,
                        'Score': score
                    })
        
        if words_df:
            words_df = pd.DataFrame(words_df)
            # 修复：去除重复条目
            words_df = words_df.drop_duplicates(['Topic', 'Word'])
            
            # 创建热力图
            pivot_df = words_df.pivot(index='Topic', columns='Word', values='Score').fillna(0)
            
            # 选择前20个词汇
            top_words = words_df.groupby('Word')['Score'].sum().nlargest(20).index
            pivot_df = pivot_df[top_words]
            
            plt.figure(figsize=(20, 12))
            sns.heatmap(pivot_df, annot=True, cmap='YlOrRd', fmt='.2f', cbar_kws={'label': 'Score'})
            plt.title('主题关键词热力图', fontsize=16, fontweight='bold')
            plt.xlabel('关键词', fontsize=12)
            plt.ylabel('主题', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # 保存图片
            plt.savefig(os.path.join(viz_dir, "topic_keywords_heatmap.png"), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(data_image_dir, "topic_keywords_heatmap_high_quality.png"), dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ 主题关键词热力图已保存到: results/high_quality_results/visualizations/topic_keywords_heatmap.png 和 data/image/topic_keywords_heatmap_high_quality.png")
        else:
            print("✗ 没有有效的主题词汇数据")
    except Exception as e:
        print(f"✗ 主题关键词热力图创建失败: {e}")
        import traceback
        traceback.print_exc()

    # 3. 文档分布饼图
    print("创建文档分布饼图...")
    total_docs = len(results_df)
    outlier_docs = len(results_df[results_df['topic'] == -1])
    valid_docs = total_docs - outlier_docs
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    labels = ['有效文档', '异常值文档']
    sizes = [valid_docs, outlier_docs]
    colors = ['lightgreen', 'lightcoral']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('文档分布', fontsize=14, fontweight='bold')
    plt.subplot(1, 2, 2)
    valid_topics = topic_info[topic_info['Topic'] != -1]
    topic_sizes = valid_topics['Count'].values
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
    plt.savefig(f'{data_image_dir}/topic_size_distribution_high_quality.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 文档分布饼图已保存到: {viz_dir}/document_distribution.png 和 {data_image_dir}/topic_size_distribution_high_quality.png")

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
    plt.savefig(f'{data_image_dir}/topic_probability_distribution_high_quality.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 主题概率分布图已保存到: {viz_dir}/topic_probability_distribution.png 和 {data_image_dir}/topic_probability_distribution_high_quality.png")

    # 5. 文档分布UMAP降维散点图（与main.ipynb风格一致）
    print("创建文档分布UMAP降维散点图...")
    try:
        from umap import UMAP
        reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(results_df[[col for col in results_df.columns if col.startswith('embedding_')]].values)
    except Exception:
        # 若未保存embedding列，则跳过
        reduced_embeddings = None
    if reduced_embeddings is not None and reduced_embeddings.shape[0] == len(results_df):
        plt.figure(figsize=(12, 8))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=results_df['topic'], cmap='tab20', alpha=0.6, s=20)
        plt.title('文档分布（UMAP降维）', fontsize=16, fontweight='bold')
        plt.xlabel('UMAP-1', fontsize=12)
        plt.ylabel('UMAP-2', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{data_image_dir}/document_distribution_umap_high_quality.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 文档分布UMAP降维散点图已保存到: {data_image_dir}/document_distribution_umap_high_quality.png")
    else:
        print("未能生成UMAP降维散点图（缺少embedding列）")

    # ====== 保存BERTopic交互式可视化为PNG到data/image ======
    print("保存BERTopic交互式可视化为PNG...")
    try:
        import plotly
        # 1. 主题分布柱状图
        try:
            fig = topic_model.visualize_barchart()
            fig.write_image(f"{data_image_dir}/topic_barchart_high_quality.png")
            print(f"✓ 交互式主题柱状图已保存: {data_image_dir}/topic_barchart_high_quality.png")
        except Exception as e:
            print(f"保存主题柱状图失败: {e}")
        # 2. 主题内容可视化
        try:
            fig = topic_model.visualize_topics()
            fig.write_image(f"{data_image_dir}/topic_keywords_heatmap_high_quality.png")
            print(f"✓ 交互式主题关键词热力图已保存: {data_image_dir}/topic_keywords_heatmap_high_quality.png")
        except Exception as e:
            print(f"保存主题关键词热力图失败: {e}")
        # 3. 层次聚类
        try:
            fig = topic_model.visualize_hierarchy()
            fig.write_image(f"{data_image_dir}/topic_hierarchy_high_quality.png")
            print(f"✓ 交互式主题层次聚类图已保存: {data_image_dir}/topic_hierarchy_high_quality.png")
        except Exception as e:
            print(f"保存主题层次聚类图失败: {e}")
        # 4. 主题相似度热力图
        try:
            fig = topic_model.visualize_heatmap()
            fig.write_image(f"{data_image_dir}/topic_similarity_high_quality.png")
            print(f"✓ 交互式主题相似度热力图已保存: {data_image_dir}/topic_similarity_high_quality.png")
        except Exception as e:
            print(f"保存主题相似度热力图失败: {e}")
        # 5. 主题随时间变化
        try:
            timestamps = None
            if texts is not None:
                try:
                    with open('data/时间.txt', 'r', encoding='utf-8') as f:
                        timestamps = [line.strip() for line in f]
                    if len(timestamps) != len(texts):
                        print(f"警告：时间戳数量({len(timestamps)})与文本数量({len(texts)})不一致，跳过topics_over_time图保存")
                        timestamps = None
                except Exception as e:
                    print(f"加载时间戳失败: {e}")
                    timestamps = None
            if texts is not None and timestamps is not None:
                topics_over_time = topic_model.topics_over_time(texts, timestamps, global_tuning=False, evolution_tuning=False)
                fig = topic_model.visualize_topics_over_time(topics_over_time)
                fig.write_image(f"{data_image_dir}/topic_over_time_high_quality.png")
                print(f"✓ 交互式主题随时间变化图已保存: {data_image_dir}/topic_over_time_high_quality.png")
        except Exception as e:
            print(f"保存主题随时间变化图失败: {e}")
    except ImportError:
        print("未安装plotly/kaleido，无法保存交互式可视化为PNG。请先 pip install plotly kaleido")
    # ====== END ======
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
- 适度文本清洗：移除所有英文单词、数字、方括号内容
- 扩展停用词：过滤更多无意义的中文词汇
- 严格分词过滤：只保留有意义的中文词汇
- 优化参数配置：提高最小文档频率，减少词汇表大小
- 增强聚类参数：降低最小主题大小，获得更稳定的主题

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
    print("改进版BERTopic主题建模工具（适度文本清洗）")
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
    viz_dir = create_visualizations(topic_model, topic_info, results_df, output_dir, texts=texts[:15000])
    
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