#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手动主题提取工具
基于BERTopic聚类结果，手动实现主题关键词提取
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from bertopic import BERTopic
from umap import UMAP
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import math

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置plotly中文字体
import plotly.io as pio

pio.templates.default = "plotly_white"


def setup_logging():
    """设置日志"""
    log_dir = "data/log"
    os.makedirs(log_dir, exist_ok=True)

    log_file = f"{log_dir}/05_4_manual_topic_extraction_log.txt"
    print(f"日志文件: {log_file}")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 重定向输出到日志文件
    import sys
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

    sys.stdout = Logger(log_file)
    return log_file


def clean_text(text):
    """清洗文本，只保留中文词汇"""
    if not text or not isinstance(text, str):
        return ""

    # 只保留中文字符和空格
    import re
    chinese_pattern = re.compile(r'[^\u4e00-\u9fff\s]')
    cleaned = chinese_pattern.sub('', text)

    # 去除多余空格
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned


def load_and_preprocess_data():
    """加载和预处理数据"""
    print("\n=== 加载和预处理数据 ===")

    # 加载嵌入向量
    embeddings = np.load('embedding/emb.npy')
    print(f"嵌入向量形状: {embeddings.shape}")

    # 加载已分词文本
    with open('data/切词.txt', 'r', encoding='utf-8') as f:
        segmented_texts = [line.strip() for line in f]
    print(f"已分词文本数量: {len(segmented_texts):,}")

    # 清洗文本：只保留中文词汇
    print("清洗文本：只保留中文词汇...")
    cleaned_texts = []
    for text in segmented_texts:
        cleaned = clean_text(text)
        cleaned_texts.append(cleaned)

    # 过滤掉清洗后为空的文本
    filtered_data = [(text, emb) for text, emb in zip(cleaned_texts, embeddings) if text.strip()]
    filtered_texts, filtered_embeddings = zip(*filtered_data)
    filtered_texts = list(filtered_texts)
    filtered_embeddings = np.array(filtered_embeddings)

    print(f"清洗后文本数量: {len(filtered_texts):,}")
    print(f"过滤后嵌入向量形状: {filtered_embeddings.shape}")

    # 统计清洗效果
    total_words = sum(len(text.split()) for text in filtered_texts)
    unique_words = len(set(word for text in filtered_texts for word in text.split()))
    avg_words = total_words / len(filtered_texts) if filtered_texts else 0

    print(f"清洗比例: {len(filtered_texts) / len(segmented_texts) * 100:.1f}%")
    print(f"清洗后总词汇数: {total_words:,}")
    print(f"清洗后唯一词汇数: {unique_words:,}")
    print(f"平均每文档词汇数: {avg_words:.1f}")

    # 显示清洗示例
    print("\n清洗示例:")
    for i in range(min(3, len(segmented_texts))):
        print(f"原文{i + 1}: {segmented_texts[i][:100]}...")
        print(f"清洗后: {filtered_texts[i][:100]}...")
        print()

    return filtered_texts, filtered_embeddings


def perform_clustering(texts, embeddings, sample_size=1000):
    """执行聚类"""
    print(f"\n=== 执行聚类 ===")
    print(f"使用前 {sample_size:,} 个样本进行聚类")

    # 取样本
    sample_texts = texts[:sample_size]
    sample_embeddings = embeddings[:sample_size]

    # 创建BERTopic模型（只用于聚类，不进行主题提取）
    topic_model = BERTopic(
        min_topic_size=10,
        nr_topics=None,
        verbose=True,
        calculate_probabilities=False  # 不计算概率
    )

    # 手动执行聚类步骤，跳过主题提取
    print("开始聚类...")

    # 1. 降维
    print("1. 降维...")
    reduced_embeddings = topic_model.umap_model.fit_transform(sample_embeddings)
    print(f"   降维后形状: {reduced_embeddings.shape}")

    # 2. 聚类
    print("2. 聚类...")
    clusters = topic_model.hdbscan_model.fit_predict(reduced_embeddings)

    # 3. 手动创建topics数组（跳过主题提取）
    topics = clusters.copy()

    # 统计聚类结果
    topic_counts = Counter(topics)
    print(f"\n聚类结果统计:")
    print(f"总聚类数: {len(topic_counts)}")
    print(f"异常值聚类(-1)文档数: {topic_counts.get(-1, 0)}")
    print(f"正常聚类数: {len([t for t in topic_counts.keys() if t != -1])}")

    # 显示前10个聚类的文档数量
    print("\n前10个聚类的文档数量:")
    for topic_id, count in topic_counts.most_common(10):
        if topic_id != -1:
            print(f"  聚类{topic_id}: {count} 个文档")
        else:
            print(f"  异常值聚类: {count} 个文档")

    return topics, topic_model, sample_texts


def extract_topic_keywords(texts, topics, top_n=20):
    """手动提取主题关键词"""
    print(f"\n=== 手动提取主题关键词 ===")

    # 创建CountVectorizer
    vectorizer = CountVectorizer(
        token_pattern=r"\S+",  # 匹配任何非空白字符序列
        min_df=2,  # 最小文档频率
        max_df=0.95,  # 最大文档频率
        max_features=5000
    )

    # 按聚类分组文档
    topic_documents = {}
    for topic_id, text in zip(topics, texts):
        if topic_id != -1:  # 跳过异常值聚类
            if topic_id not in topic_documents:
                topic_documents[topic_id] = []
            topic_documents[topic_id].append(text)

    print(f"发现 {len(topic_documents)} 个正常聚类")

    # 为每个聚类提取关键词
    topic_keywords = {}
    topic_info = []

    for topic_id, docs in topic_documents.items():
        if len(docs) < 2:  # 跳过文档太少的聚类
            continue

        # 合并文档
        merged_doc = " ".join(docs)

        try:
            # 使用TF-IDF提取关键词
            tfidf = TfidfVectorizer(
                token_pattern=r"\S+",
                min_df=1,  # 改为1，避免与max_df冲突
                max_df=1.0,  # 改为1.0，避免与min_df冲突
                max_features=1000
            )

            tfidf_matrix = tfidf.fit_transform([merged_doc])
            feature_names = tfidf.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]

            # 获取top_n关键词
            top_indices = np.argsort(tfidf_scores)[::-1][:top_n]
            keywords = [(feature_names[i], tfidf_scores[i]) for i in top_indices if tfidf_scores[i] > 0]

            topic_keywords[topic_id] = keywords

            # 记录主题信息
            topic_info.append({
                'Topic': topic_id,
                'Count': len(docs),
                'Keywords': ', '.join([word for word, score in keywords[:10]]),
                'Top_Keywords': keywords[:10]
            })

            print(f"聚类{topic_id}: {len(docs)}个文档, 关键词: {', '.join([word for word, score in keywords[:5]])}")

        except Exception as e:
            print(f"聚类{topic_id}关键词提取失败: {e}")
            continue

    return topic_keywords, topic_info


def create_results_dataframe(texts, topics, topic_keywords):
    """创建结果数据框"""
    print(f"\n=== 创建结果数据框 ===")

    results = []
    for i, (text, topic_id) in enumerate(zip(texts, topics)):
        # 获取主题关键词
        if topic_id in topic_keywords:
            keywords = topic_keywords[topic_id]
            top_keywords = ', '.join([word for word, score in keywords[:5]])
        else:
            top_keywords = "异常值"

        results.append({
            'Document_ID': i,
            'Topic': topic_id,
            'Text': text,
            'Top_Keywords': top_keywords
        })

    results_df = pd.DataFrame(results)
    print(f"结果数据框形状: {results_df.shape}")
    print(f"主题分布:\n{results_df['Topic'].value_counts().head(10)}")

    return results_df


def save_results(results_df, topic_info, output_dir="results/5_4"):
    """保存结果"""
    print(f"\n=== 保存结果 ===")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ 创建输出目录: {output_dir}")

    # 保存主题建模结果
    results_file = f"{output_dir}/topic_modeling_results.csv"
    results_df.to_csv(results_file, index=False, encoding='utf-8')
    print(f"✓ 主题建模结果已保存到: {results_file}")

    # 保存主题信息
    topic_info_df = pd.DataFrame(topic_info)
    topic_info_file = f"{output_dir}/topic_info.csv"
    topic_info_df.to_csv(topic_info_file, index=False, encoding='utf-8')
    print(f"✓ 主题信息已保存到: {topic_info_file}")

    # 兼容main.ipynb风格的保存
    csv1 = "data/聚类结果.csv"
    results_df.to_csv(csv1, index=False, encoding='utf-8')
    print(f"✓ 聚类结果已保存到: {csv1}")

    # 保存聚类结果2.csv（含原文和时间）
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

    return output_dir


def create_topic_word_barcharts(topic_info, viz_dir, top_n=10):
    # 只展示前16个主题
    show_topics = topic_info[:8]
    ncols = 4
    nrows = 2
    plt.figure(figsize=(5 * ncols, 3.5 * nrows))
    for idx, info in enumerate(show_topics):
        keywords = info['Top_Keywords'][:top_n]
        words = [w for w, s in keywords][::-1]
        scores = [s for w, s in keywords][::-1]
        ax = plt.subplot(nrows, ncols, idx + 1)
        ax.barh(words, scores, color=plt.cm.Set3(idx % 12))
        ax.set_title(info['Top_Keywords'][0][0] if info['Top_Keywords'] else f"主题{info['Topic']}", fontsize=14,
                     fontweight='bold')
        ax.set_xlabel('分数', fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='x', labelsize=10)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    plt.suptitle("各主题关键词分数（前16个主题）", fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(viz_dir, "1_topic_word_barcharts.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 主题关键词分数条形图已保存")


def create_topic_barchart(topic_info, viz_dir):
    topic_counts = [info['Count'] for info in topic_info]
    topic_labels = [info['Top_Keywords'][0][0] if info['Top_Keywords'] else f"主题{info['Topic']}" for info in
                    topic_info]
    plt.figure(figsize=(18, 8))
    bars = plt.bar(range(len(topic_counts)), topic_counts, color='skyblue', alpha=0.7)
    for bar, count in zip(bars, topic_counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(count), ha='center', va='bottom',
                 fontsize=10)
    plt.xlabel('主题关键词', fontsize=14, fontweight='bold')
    plt.ylabel('文档数量', fontsize=14, fontweight='bold')
    plt.title('主题分布柱状图', fontsize=18, fontweight='bold')
    plt.xticks(range(len(topic_labels)), topic_labels, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "2_topic_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 主题分布柱状图已保存")


def create_topic_visualization(topic_info, embeddings, topics, viz_dir):
    try:
        umap_model = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=42)
        reduced_embeddings = umap_model.fit_transform(embeddings)
        plt.figure(figsize=(14, 12))
        unique_topics = sorted(set(topics))
        topic_id2label = {}
        for info in topic_info:
            topic_id2label[info['Topic']] = info['Top_Keywords'][0][0] if info[
                'Top_Keywords'] else f"主题{info['Topic']}"
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_topics)))
        for i, topic_id in enumerate(unique_topics):
            mask = topics == topic_id
            label = topic_id2label.get(topic_id, f"主题{topic_id}")
            if topic_id == -1:
                plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], c='gray', alpha=0.3, s=20,
                            label='异常值')
            else:
                plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], c=[colors[i]], alpha=0.7, s=30,
                            label=label)
        plt.xlabel('UMAP Component 1', fontsize=14)
        plt.ylabel('UMAP Component 2', fontsize=14)
        plt.title('主题分布可视化', fontsize=18, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "3_topic_visualization.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 主题可视化已保存")
    except Exception as e:
        print(f"✗ 主题可视化创建失败: {e}")


def create_document_visualization(texts, embeddings, topics, viz_dir, topic_info):
    try:
        umap_model = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine', random_state=42)
        reduced_embeddings = umap_model.fit_transform(embeddings)
        plt.figure(figsize=(16, 12))
        unique_topics = sorted(set(topics))
        topic_id2label = {}
        for info in topic_info:
            topic_id2label[info['Topic']] = info['Top_Keywords'][0][0] if info[
                'Top_Keywords'] else f"主题{info['Topic']}"
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_topics)))
        for i, topic_id in enumerate(unique_topics):
            mask = topics == topic_id
            label = topic_id2label.get(topic_id, f"主题{topic_id}")
            if topic_id == -1:
                plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], c='lightgray', alpha=0.4, s=15,
                            label='异常值')
            else:
                plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], c=[colors[i]], alpha=0.6, s=20,
                            label=label)
        plt.xlabel('UMAP Component 1', fontsize=14)
        plt.ylabel('UMAP Component 2', fontsize=14)
        plt.title('文档分布可视化', fontsize=18, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "4_document_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 文档分布可视化已保存")
    except Exception as e:
        print(f"✗ 文档分布可视化创建失败: {e}")


def create_hierarchical_visualization(topic_info, viz_dir):
    try:
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import pdist
        topic_sizes = [info['Count'] for info in topic_info]
        topic_labels = [info['Top_Keywords'][0][0] if info['Top_Keywords'] else f"主题{info['Topic']}" for info in
                        topic_info]
        if len(topic_sizes) > 1:
            normalized_sizes = np.array(topic_sizes).reshape(-1, 1)
            normalized_sizes = (normalized_sizes - normalized_sizes.mean()) / normalized_sizes.std()
            linkage_matrix = linkage(normalized_sizes, method='ward')
            plt.figure(figsize=(14, 8))
            dendrogram(linkage_matrix, labels=topic_labels, leaf_rotation=90, leaf_font_size=12)
            plt.xlabel('主题', fontsize=14, fontweight='bold')
            plt.ylabel('距离', fontsize=14, fontweight='bold')
            plt.title('主题层次聚类树状图', fontsize=18, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "5_topic_hierarchy.png"), dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ 主题层次聚类可视化已保存")
        else:
            print("⚠ 主题数量不足，跳过层次聚类可视化")
    except Exception as e:
        print(f"✗ 层次聚类可视化创建失败: {e}")


def create_topic_keywords_heatmap(topic_info, viz_dir):
    try:
        words_data = []
        for info in topic_info:
            topic_id = info['Topic']
            for word, score in info['Top_Keywords']:
                words_data.append(
                    {'Topic': info['Top_Keywords'][0][0] if info['Top_Keywords'] else f"主题{topic_id}", 'Word': word,
                     'Score': score})
        if words_data:
            words_df = pd.DataFrame(words_data)
            top_words = words_df.groupby('Word')['Score'].sum().nlargest(15).index
            words_df = words_df[words_df['Word'].isin(top_words)]
            pivot_df = words_df.pivot_table(index='Topic', columns='Word', values='Score', aggfunc='mean').fillna(0)
            plt.figure(figsize=(18, 10))
            sns.heatmap(pivot_df, annot=True, cmap='YlOrRd', fmt='.2f', cbar_kws={'label': 'TF-IDF Score'},
                        linewidths=0.5)
            plt.title('主题关键词热力图', fontsize=16, fontweight='bold')
            plt.xlabel('关键词', fontsize=12, fontweight='bold')
            plt.ylabel('主题', fontsize=12, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "6_topic_keywords_heatmap.png"), dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ 主题关键词热力图已保存")
    except Exception as e:
        print(f"✗ 主题关键词热力图创建失败: {e}")


def create_topic_probability_distribution(topic_info, viz_dir):
    try:
        topic_counts = [info['Count'] for info in topic_info]
        topic_labels = [info['Top_Keywords'][0][0] if info['Top_Keywords'] else f"主题{info['Topic']}" for info in
                        topic_info]
        total_docs = sum(topic_counts)
        probabilities = [count / total_docs for count in topic_counts]
        plt.figure(figsize=(14, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(topic_counts)))
        wedges, texts, autotexts = plt.pie(probabilities, labels=topic_labels, autopct='%1.1f%%', colors=colors,
                                           startangle=90, pctdistance=0.85)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        plt.title('主题概率分布', fontsize=18, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "7_topic_probability_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 主题概率分布图已保存")
    except Exception as e:
        print(f"✗ 主题概率分布图创建失败: {e}")


def create_topic_similarity_visualization(topic_info, embeddings, topics, viz_dir):
    try:
        topic_centers = {}
        topic_labels = []
        for info in topic_info:
            topic_id = info['Topic']
            mask = topics == topic_id
            if mask.sum() > 0:
                topic_centers[topic_id] = embeddings[mask].mean(axis=0)
                if info['Top_Keywords']:
                    topic_labels.append(info['Top_Keywords'][0][0])
                else:
                    topic_labels.append(f"主题{topic_id}")
        if len(topic_centers) > 1:
            topic_ids = list(topic_centers.keys())
            similarity_matrix = np.zeros((len(topic_ids), len(topic_ids)))
            for i, tid1 in enumerate(topic_ids):
                for j, tid2 in enumerate(topic_ids):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        center1 = topic_centers[tid1]
                        center2 = topic_centers[tid2]
                        similarity = np.dot(center1, center2) / (np.linalg.norm(center1) * np.linalg.norm(center2))
                        similarity_matrix[i, j] = similarity
            plt.figure(figsize=(24, 20))
            sns.heatmap(similarity_matrix, annot=True, cmap='Reds', center=0.5, xticklabels=topic_labels,
                        yticklabels=topic_labels, fmt='.1f', annot_kws={"size": 8})
            plt.title('主题相似度矩阵', fontsize=24, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "8_topic_similarity.png"), dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ 主题相似度可视化已保存")
        else:
            print("⚠ 主题数量不足，跳过主题相似度可视化")
    except Exception as e:
        print(f"✗ 主题相似度可视化创建失败: {e}")


def create_topic_size_distribution(topic_info, viz_dir):
    try:
        topic_counts = [info['Count'] for info in topic_info]
        plt.figure(figsize=(12, 6))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.hist(topic_counts, bins=min(10, len(topic_counts)), alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('主题大小', fontweight='bold')
        ax1.set_ylabel('频次', fontweight='bold')
        ax1.set_title('主题大小分布直方图', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax2.boxplot(topic_counts, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
        ax2.set_ylabel('主题大小', fontweight='bold')
        ax2.set_title('主题大小分布箱线图', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "9_topic_size_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 主题大小分布图已保存")
    except Exception as e:
        print(f"✗ 主题大小分布图创建失败: {e}")


def create_topics_over_time_visualization(topic_info, viz_dir):
    try:
        timestamps = []
        try:
            with open('data/时间.txt', 'r', encoding='utf-8') as f:
                timestamps = [line.strip() for line in f if line.strip()]
        except:
            print("⚠ 无法加载时间数据，跳过动态主题可视化")
            return
        if not timestamps:
            print("⚠ 时间数据为空，跳过动态主题可视化")
            return
        import re
        months = []
        for t in timestamps:
            m = re.match(r'(\d{4})[-/]?(\d{2})', t)
            if m:
                months.append(f"{m.group(1)}-{m.group(2)}")
            else:
                months.append(t)
        from collections import Counter
        month_counts = Counter(months)
        sorted_months = sorted(month_counts.keys())
        counts = [month_counts[m] for m in sorted_months]
        plt.figure(figsize=(16, 8))
        plt.plot(sorted_months, counts, marker='o', linewidth=2, markersize=6, color='blue')
        plt.fill_between(sorted_months, counts, alpha=0.3, color='lightblue')
        plt.xlabel('时间（月）', fontweight='bold')
        plt.ylabel('文档数量', fontweight='bold')
        plt.title('文档时间分布（按月）', fontsize=20, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "10_topics_over_time.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 动态主题模型可视化已保存")
    except Exception as e:
        print(f"✗ 动态主题模型可视化创建失败: {e}")


def create_visualizations(topic_info, output_dir, texts=None, embeddings=None, topics=None):
    """创建完整的可视化（参考main.ipynb）"""
    print(f"\n=== 创建完整可视化 ===")

    # 创建可视化目录
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # 1. 主题分布柱状图 (参考main.ipynb的visualize_barchart)
    print("1. 创建主题分布柱状图...")
    create_topic_barchart(topic_info, viz_dir)

    # 2. 主题可视化 (参考main.ipynb的visualize_topics)
    print("2. 创建主题可视化...")
    if embeddings is not None and topics is not None:
        create_topic_visualization(topic_info, embeddings, topics, viz_dir)

    # 3. 文档分布可视化 (参考main.ipynb的visualize_documents)
    print("3. 创建文档分布可视化...")
    if texts is not None and embeddings is not None:
        create_document_visualization(texts, embeddings, topics, viz_dir, topic_info)

    # 4. 层次聚类可视化 (参考main.ipynb的visualize_hierarchy)
    print("4. 创建层次聚类可视化...")
    create_hierarchical_visualization(topic_info, viz_dir)

    # 5. 主题关键词热力图
    print("5. 创建主题关键词热力图...")
    create_topic_keywords_heatmap(topic_info, viz_dir)

    # 6. 主题概率分布图
    print("6. 创建主题概率分布图...")
    create_topic_probability_distribution(topic_info, viz_dir)

    # 7. 主题相似度图
    print("7. 创建主题相似度图...")
    if embeddings is not None and topics is not None:
        create_topic_similarity_visualization(topic_info, embeddings, topics, viz_dir)

    # 8. 主题大小分布图
    print("8. 创建主题大小分布图...")
    create_topic_size_distribution(topic_info, viz_dir)

    # 9. 动态主题模型可视化 (参考main.ipynb的topics_over_time)
    print("9. 创建动态主题模型可视化...")
    create_topics_over_time_visualization(topic_info, viz_dir)

    # 10. 主题关键词分数条形图（对齐BERTopic官方）
    print("10. 创建主题关键词分数条形图...")
    create_topic_word_barcharts(topic_info, viz_dir, top_n=10)

    print("✓ 所有可视化完成")


def generate_report(topic_info, results_df, output_dir):
    """生成报告"""
    print(f"\n=== 生成报告 ===")
    report_file = os.path.join(output_dir, "topic_analysis_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("主题提取分析报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总文档数: {len(results_df):,}\n")
        f.write(f"主题数量: {len(topic_info)}\n")
        f.write(f"异常值文档数: {len(results_df[results_df['Topic'] == -1])}\n\n")
        f.write("主题详细信息:\n")
        f.write("-" * 30 + "\n")
        for info in topic_info:
            f.write(f"主题 {info['Topic']}:\n")
            f.write(f"  文档数量: {info['Count']}\n")
            f.write(f"  关键词: {info['Keywords']}\n\n")
        f.write("主题分布统计:\n")
        f.write("-" * 30 + "\n")
        topic_counts = results_df['Topic'].value_counts()
        for topic_id, count in topic_counts.head(10).items():
            if topic_id != -1:
                f.write(f"主题 {topic_id}: {count:,} 个文档\n")
            else:
                f.write(f"异常值: {count:,} 个文档\n")
    print(f"✓ 报告已保存到: {report_file}")


def main():
    """主函数"""
    print("=" * 60)
    print("=" * 60)
    print("手动主题提取工具")
    print("=" * 60)
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 设置日志
    setup_logging()

    try:
        # 1. 加载和预处理数据
        texts, embeddings = load_and_preprocess_data()

        # 2. 执行聚类
        topics, topic_model, sample_texts = perform_clustering(texts, embeddings, sample_size=1000)

        # 3. 手动提取主题关键词
        topic_keywords, topic_info = extract_topic_keywords(sample_texts, topics)

        # 4. 创建结果数据框
        results_df = create_results_dataframe(sample_texts, topics, topic_keywords)

        # 5. 保存结果
        output_dir = save_results(results_df, topic_info)

        # 6. 创建完整可视化（包含main.ipynb中的所有可视化）
        create_visualizations(topic_info, output_dir, sample_texts, embeddings[:len(sample_texts)], topics)

        # 7. 生成报告
        generate_report(topic_info, results_df, output_dir)

        print(f"\n✓ 手动主题提取完成！")
        print(f"✓ 发现 {len(topic_info)} 个主题")
        print(f"✓ 结果保存在: {output_dir}")
        print(f"✓ 可视化结果保存在: {output_dir}/visualizations/")

    except Exception as e:
        print(f"✗ 执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()