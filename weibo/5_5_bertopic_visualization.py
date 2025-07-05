#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERTopic官方可视化脚本 (5-5)
使用BERTopic的官方可视化API，基于5-4的手动主题提取结果
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
import plotly.io as pio
pio.renderers.default = "png"

def main():
    print("=== BERTopic官方可视化 (5-5) ===")
    
    # 1. 读取5-4的结果
    print("1. 读取5-4的结果...")
    
    # 读取主题信息
    import pandas as pd
    topic_info = pd.read_csv("results/5_4/topic_info.csv")
    topic_modeling_results = pd.read_csv("results/5_4/topic_modeling_results.csv")
    
    # 读取原始数据
    weibo_data = pd.read_csv("data/weibo_clean_data.csv")
    
    # 加载embeddings (用于文档分布可视化)
    embeddings = np.load("embedding/emb.npy")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # 2. 准备数据
    print("2. 准备数据...")
    
    # 获取有效的主题和文档
    valid_topics = topic_modeling_results['Topic'].tolist()
    valid_texts = weibo_data['clean_txt'].tolist()
    
    # 过滤掉空文档
    valid_data = [(text, topic) for text, topic in zip(valid_texts, valid_topics) 
                  if isinstance(text, str) and text.strip()]
    
    texts = [item[0] for item in valid_data]
    topics = [item[1] for item in valid_data]
    
    # 过滤embeddings以匹配有效文档
    valid_indices = [i for i, (text, topic) in enumerate(zip(valid_texts, valid_topics)) 
                    if isinstance(text, str) and text.strip()]
    embeddings = embeddings[valid_indices]
    
    print(f"有效文档数量: {len(texts)}")
    print(f"主题数量: {len(set(topics))}")
    print(f"过滤后embeddings shape: {embeddings.shape}")
    
    # 3. 构建BERTopic对象
    print("3. 构建BERTopic对象...")
    
    topic_model = BERTopic(
        embedding_model=None,  # 使用预计算的embeddings
        umap_model=UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        ),
        hdbscan_model=HDBSCAN(
            min_cluster_size=15,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        ),
        vectorizer_model=CountVectorizer(
            token_pattern=r'(?u)\b\w+\b',  # 支持中文的token pattern
            min_df=1,
            max_df=1.0,
            ngram_range=(1, 2)
        ),
        verbose=True
    )
    
    # 4. 手动构建BERTopic对象
    print("4. 手动构建BERTopic对象...")
    
    # 设置主题
    topic_model.topics_ = topics
    
    # 读取分词文本
    with open('data/切词.txt', 'r', encoding='utf-8') as f:
        tokenized_texts = [line.strip() for line in f]

    # 合并每个主题的分词文本
    merged_docs = []
    for topic in sorted(set(topics)):
        indices = [i for i, t in enumerate(topics) if t == topic]
        merged = ' '.join([tokenized_texts[i] for i in indices])
        merged_docs.append(merged)
    
    # 使用CountVectorizer提取词汇
    vectorizer = CountVectorizer(
        token_pattern=r'(?u)\b\w+\b',
        min_df=1,
        max_df=1.0,
        ngram_range=(1, 2)
    )

    try:
        X = vectorizer.fit_transform(merged_docs)
        topic_model.c_tf_idf_ = X
        topic_model.vectorizer_model = vectorizer
        topic_model.words_ = vectorizer.get_feature_names_out()
        # 新增：补充ctfidf_model的fit，修复层次聚类报错
        from bertopic.vectorizers import ClassTfidfTransformer
        ctfidf_model = ClassTfidfTransformer()
        ctfidf_model.fit(X)
        topic_model.ctfidf_model = ctfidf_model
    except Exception as e:
        print(f"CountVectorizer异常: {e}")
        return
    
    # 构建topic_representations_（主题关键词表示）
    print('构建topic_representations_...')
    topic_model.topic_representations_ = {}
    for topic in set(topics):
        if topic == -1:  # 跳过噪声主题
            continue
        
        # 获取该主题在c_tf_idf中的索引
        topic_idx = list(sorted(set(topics))).index(topic)
        
        # 获取该主题的词汇权重
        topic_weights = topic_model.c_tf_idf_[topic_idx].toarray().flatten()
        
        # 获取前20个最重要的词汇
        top_indices = np.argsort(topic_weights)[::-1][:20]
        topic_words = []
        
        for idx in top_indices:
            if topic_weights[idx] > 0:
                word = topic_model.words_[idx]
                weight = float(topic_weights[idx])
                # 只保留1-3字的关键词
                if 1 <= len(word) <= 3:
                    topic_words.append((word, weight))
        
        # 确保topic_representations_的格式正确
        topic_model.topic_representations_[topic] = topic_words
    
    # 确保异常值主题 -1 也有一个空关键词列表
    if -1 not in topic_model.topic_representations_:
        topic_model.topic_representations_[-1] = []
    
    # 设置其他必要属性
    topic_model.topic_sizes_ = {topic: len([t for t in topics if t == topic]) for topic in set(topics)}
    
    # 设置topic_embeddings_（用于文档可视化）
    print('设置topic_embeddings_...')
    topic_embeddings = []
    for topic in sorted(set(topics)):
        topic_doc_indices = [i for i, t in enumerate(topics) if t == topic]
        if topic_doc_indices:
            topic_emb = embeddings[topic_doc_indices].mean(axis=0)
            topic_embeddings.append(topic_emb)
        else:
            # 没有文档的主题（如-1），用零向量
            topic_embeddings.append(np.zeros(embeddings.shape[1]))
    topic_model.topic_embeddings_ = np.array(topic_embeddings)
    print(f'topic_embeddings_ shape: {topic_model.topic_embeddings_.shape}')
    
    print(f"成功构建BERTopic对象，包含 {len(topic_model.topic_representations_)} 个主题")
    
    # 读取主题-关键词映射
    topic2label = {}
    for _, row in topic_info.iterrows():
        topic = row['Topic']
        keywords = row['Top_Keywords']
        if isinstance(keywords, str) and keywords.strip():
            # 解析Top_Keywords列，提取第一个关键词
            # 格式: "[('关键词', 分数), ('关键词', 分数), ...]"
            try:
                # 去除开头的 "[(" 和结尾的 ")]"
                clean_keywords = keywords.strip()
                if clean_keywords.startswith("[(") and clean_keywords.endswith(")]"):
                    clean_keywords = clean_keywords[2:-2]  # 去除 "[(" 和 ")]"
                
                # 分割关键词对
                keyword_pairs = clean_keywords.split("), (")
                if keyword_pairs:
                    # 取第一个关键词对
                    first_pair = keyword_pairs[0]
                    # 提取关键词（去除引号和空格）
                    label = first_pair.split("',")[0].replace("'", "").replace('"', '').strip()
                    
                    # 如果第一个关键词太长，尝试找更短的（限制为3个字符以内）
                    if len(label) > 3:
                        for pair in keyword_pairs:
                            word = pair.split("',")[0].replace("'", "").replace('"', '').strip()
                            if 1 <= len(word) <= 3:
                                label = word
                                break
                    
                    topic2label[topic] = label
                else:
                    topic2label[topic] = f"主题{topic}"
            except Exception as e:
                print(f"解析主题{topic}关键词失败: {e}")
                topic2label[topic] = f"主题{topic}"
        else:
            topic2label[topic] = f"主题{topic}"

    # 构建 custom_labels_
    custom_labels = []
    for topic in sorted(set(topics)):
        if topic == -1:
            custom_labels.append("异常值")
        else:
            custom_labels.append(topic2label.get(topic, f"主题{topic}"))
    topic_model.custom_labels_ = custom_labels
    
    # 输出主题标签长度调试信息
    print("主题标签预览：", custom_labels)
    print("最长标签长度：", max(len(str(label)) for label in custom_labels))

    # 5. 生成可视化
    print("5. 生成可视化...")
    
    # 创建结果目录
    os.makedirs("results/5_5_old_result", exist_ok=True)
    
    # 1. 主题关键词条形图 (对应main.ipynb Cell 12)
    print("生成主题关键词条形图...")
    fig = topic_model.visualize_barchart(top_n_topics=16)
    fig.write_html("results/5_5/1_topic_barchart.html")
    try:
        fig.write_image("results/5_5/1_topic_barchart.png", width=1200, height=800)
        print('✓ 1. 主题关键词条形图图片已保存')
    except Exception as e:
        print(f'⚠ 1. 主题关键词条形图图片导出失败: {e}')
    
    # 2. 主题可视化 (对应main.ipynb Cell 13)
    print("生成主题可视化...")
    fig = topic_model.visualize_topics()
    fig.write_html("results/5_5/2_topic_visualization.html")
    try:
        fig.write_image("results/5_5/2_topic_visualization.png", width=1200, height=800)
        print('✓ 2. 主题可视化图片已保存')
    except Exception as e:
        print(f'⚠ 2. 主题可视化图片导出失败: {e}')
    
    # 3. 文档分布 (对应main.ipynb Cell 14)
    print("生成文档分布...")
    # 调试：检查get_topic返回值
    for topic in sorted(set(topics)):
        result = topic_model.get_topic(topic)
        if not isinstance(result, list):
            print(f"主题 {topic} get_topic 返回异常: {result} 类型: {type(result)}")
    # 使用UMAP降维到2D用于文档可视化
    reduced_embeddings = UMAP(
        n_neighbors=10, 
        n_components=2, 
        min_dist=0.0, 
        metric='cosine'
    ).fit_transform(embeddings)
    
    fig = topic_model.visualize_documents(
        texts, 
        reduced_embeddings=reduced_embeddings, 
        hide_document_hover=True
    )
    fig.write_html("results/5_5/3_document_distribution.html")
    try:
        # 尝试多种配置来避免kaleido bug
        try:
            # 方法1：使用更简单的配置
            fig.write_image("results/5_5/3_document_distribution.png", width=1200, height=800)
            print('✓ 3. 文档分布图片已保存')
        except Exception as e1:
            print(f'方法1失败: {e1}')
            try:
                # 方法2：使用更小的尺寸
                fig.write_image("results/5_5/3_document_distribution.png", width=800, height=600)
                print('✓ 3. 文档分布图片已保存（小尺寸）')
            except Exception as e2:
                print(f'方法2失败: {e2}')
                try:
                    # 方法3：使用默认尺寸
                    fig.write_image("results/5_5/3_document_distribution.png")
                    print('✓ 3. 文档分布图片已保存（默认尺寸）')
                except Exception as e3:
                    print(f'方法3失败: {e3}')
                    # 方法4：尝试简化图表
                    try:
                        # 创建一个简化版本的文档分布图
                        import plotly.express as px
                        import pandas as pd
                        
                        # 创建简化的散点图
                        df = pd.DataFrame({
                            'x': reduced_embeddings[:, 0],
                            'y': reduced_embeddings[:, 1],
                            'topic': [topic2label.get(t, f"主题{t}") for t in topics]
                        })
                        
                        fig_simple = px.scatter(
                            df, x='x', y='y', color='topic',
                            title='文档分布图（简化版）',
                            labels={'x': 'UMAP 1', 'y': 'UMAP 2'}
                        )
                        fig_simple.update_layout(
                            width=1200, height=800,
                            font=dict(size=12)
                        )
                        fig_simple.write_image("results/5_5/3_document_distribution.png")
                        print('✓ 3. 文档分布图片已保存（简化版）')
                    except Exception as e4:
                        print(f'⚠ 3. 文档分布图片导出失败（所有方法都失败）: {e4}')
                        print('HTML版本仍然可用: results/5_5/3_document_distribution.html')
    except Exception as e:
        print(f'⚠ 3. 文档分布图片导出失败: {e}')
    
    # 4. 主题层次聚类 (对应main.ipynb Cell 16)
    print("生成主题层次聚类...")
    hierarchical_topics = topic_model.hierarchical_topics(texts)
    fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
    fig.write_html("results/5_5/4_topic_hierarchy.html")
    try:
        fig.write_image("results/5_5/4_topic_hierarchy.png", width=1200, height=800)
        print('✓ 4. 主题层次聚类图片已保存')
    except Exception as e:
        print(f'⚠ 4. 主题层次聚类图片导出失败: {e}')
    
    # 5. 主题相似性热力图 (BERTopic标准可视化)
    print("生成主题相似性热力图...")
    fig = topic_model.visualize_heatmap()
    fig.write_html("results/5_5/5_topic_similarity.html")
    try:
        fig.write_image("results/5_5/5_topic_similarity.png", width=1200, height=800)
        print('✓ 5. 主题相似性热力图图片已保存')
    except Exception as e:
        print(f'⚠ 5. 主题相似性热力图图片导出失败: {e}')
    
    # 6. 动态主题模型时间分布 (对应main.ipynb Cell 23)
    print("生成动态主题模型时间分布...")
    # 检查是否有时间数据
    if os.path.exists('data/时间.txt'):
        with open('data/时间.txt', 'r', encoding='utf-8') as f:
            timestamps = [int(line.strip()) for line in f if line.strip()]
        
        print(f"时间数据长度: {len(timestamps)}, 文本长度: {len(texts)}")
        
        # 如果时间数据长度不匹配，尝试截取或填充
        if len(timestamps) != len(texts):
            if len(timestamps) > len(texts):
                # 截取时间数据
                timestamps = timestamps[:len(texts)]
                print(f"截取时间数据到 {len(timestamps)} 条")
            else:
                # 用最后一个时间填充
                last_timestamp = timestamps[-1] if timestamps else 2023
                timestamps.extend([last_timestamp] * (len(texts) - len(timestamps)))
                print(f"填充时间数据到 {len(timestamps)} 条")
        
        if len(timestamps) == len(texts):
            topics_over_time = topic_model.topics_over_time(
                texts, 
                timestamps, 
                global_tuning=False, 
                evolution_tuning=False
            )
            fig = topic_model.visualize_topics_over_time(topics_over_time)
            fig.write_html("results/5_5/6_topics_over_time.html")
            try:
                fig.write_image("results/5_5/6_topics_over_time.png", width=1200, height=800)
                print('✓ 6. 动态主题模型时间分布图片已保存')
            except Exception as e:
                print(f'⚠ 6. 动态主题模型时间分布图片导出失败: {e}')
        else:
            print('⚠ 时间数据处理后仍不匹配，跳过时间分布可视化')
    else:
        print('⚠ 未找到时间数据文件，跳过时间分布可视化')
    
    # 7. 合并主题后的主题可视化 (对应main.ipynb Cell 19)
    print("生成合并主题后的主题可视化...")
    try:
        # 检查主题数量，如果主题太少就不合并
        unique_topics = set(topics)
        if len(unique_topics) <= 4:
            print(f'⚠ 主题数量太少 ({len(unique_topics)})，跳过合并主题')
        else:
            # 手动实现主题合并，避免BERTopic对象的问题
            existing_topics = sorted([t for t in unique_topics if t != -1])
            print(f"现有主题: {existing_topics}")
            
            if len(existing_topics) >= 4:
                # 根据实际主题数量调整合并配置
                if len(existing_topics) >= 20:
                    merge_config = [
                        [19, 2] if 19 in existing_topics and 2 in existing_topics else [existing_topics[0], existing_topics[1]],
                        [9, 6, 1, 12, 20, 5, 15, 18] if all(t in existing_topics for t in [9, 6, 1, 12, 20, 5, 15, 18]) else existing_topics[2:10],
                        [11, 16, 4, 10, 7] if all(t in existing_topics for t in [11, 16, 4, 10, 7]) else existing_topics[10:15],
                        [17, 13, 0, 3, 8, 14] if all(t in existing_topics for t in [17, 13, 0, 3, 8, 14]) else existing_topics[15:21]
                    ]
                else:
                    # 如果主题数量不够，分成4组
                    group_size = len(existing_topics) // 4
                    merge_config = []
                    for i in range(4):
                        start_idx = i * group_size
                        end_idx = start_idx + group_size if i < 3 else len(existing_topics)
                        if start_idx < len(existing_topics):
                            merge_config.append(existing_topics[start_idx:end_idx])
                
                print(f"合并配置: {merge_config}")
                
                # 手动实现主题合并
                merged_topics = topics.copy()
                topic_mapping = {}
                
                # 创建主题映射
                for i, group in enumerate(merge_config):
                    new_topic_id = i
                    for old_topic in group:
                        topic_mapping[old_topic] = new_topic_id
                
                # 应用映射
                for i, topic in enumerate(merged_topics):
                    if topic in topic_mapping:
                        merged_topics[i] = topic_mapping[topic]
                
                # 创建新的主题信息
                merged_topic_info = []
                for i, group in enumerate(merge_config):
                    # 合并关键词
                    all_keywords = []
                    for topic in group:
                        if topic in topic_model.topic_representations_:
                            keywords = topic_model.topic_representations_[topic]
                            all_keywords.extend(keywords)
                    
                    # 按权重排序并去重
                    keyword_dict = {}
                    for word, weight in all_keywords:
                        if 1 <= len(word) <= 3:  # 只保留1-3字
                            if word not in keyword_dict:
                                keyword_dict[word] = weight
                            else:
                                keyword_dict[word] += weight
                    
                    # 取前10个关键词
                    sorted_keywords = sorted(keyword_dict.items(), key=lambda x: x[1], reverse=True)[:10]
                    keywords_str = ", ".join([f"'{word}'" for word, _ in sorted_keywords])
                    
                    merged_topic_info.append({
                        'Topic': i,
                        'Count': sum(1 for t in merged_topics if t == i),
                        'Top_Keywords': keywords_str
                    })
                
                # 创建合并后的主题标签（限制为3个字符以内）
                merged_topic2label = {}
                for info in merged_topic_info:
                    topic = info['Topic']
                    keywords = info['Top_Keywords']
                    if isinstance(keywords, str) and keywords.strip():
                        label = keywords.split(',')[0].strip().replace("'", "").replace('"', '')
                        # 如果第一个关键词太长，尝试找更短的（限制为3个字符以内）
                        if len(label) > 3:
                            for word in keywords.split(','):
                                word = word.strip().replace("'", "").replace('"', '')
                                if 1 <= len(word) <= 3:
                                    label = word
                                    break
                        merged_topic2label[topic] = label
                    else:
                        merged_topic2label[topic] = f"主题{topic}"
                
                # 创建简化版的主题可视化
                try:
                    # 使用plotly创建合并主题后的可视化
                    fig = px.scatter(
                        x=[i for i in range(len(merged_topic_info))],
                        y=[info['Count'] for info in merged_topic_info],
                        text=[merged_topic2label.get(info['Topic'], f"主题{info['Topic']}") for info in merged_topic_info],
                        title='合并主题后的主题分布',
                        labels={'x': '主题ID', 'y': '文档数量'}
                    )
                    fig.update_traces(textposition='top center')
                    fig.update_layout(
                        width=1200, height=800,
                        font=dict(size=12)
                    )
                    
                    fig.write_html("results/5_5/7_merged_topics_visualization.html")
                    try:
                        fig.write_image("results/5_5/7_merged_topics_visualization.png", width=1200, height=800)
                        print('✓ 7. 合并主题后的主题可视化图片已保存')
                    except Exception as e:
                        print(f'⚠ 7. 合并主题后的主题可视化图片导出失败: {e}')
                        
                except Exception as e:
                    print(f'⚠ 创建合并主题可视化失败: {e}')
            else:
                print(f'⚠ 主题数量不足，跳过合并主题')
    except Exception as e:
        print(f'⚠ 合并主题失败，跳过合并主题可视化: {e}')
    
    # 8. 合并主题后的文档分布 (对应main.ipynb Cell 20)
    print("生成合并主题后的文档分布...")
    try:
        # 检查是否成功合并了主题
        if 'merged_topics' in locals() and 'merged_topic2label' in locals():
            print(f"主题已合并: {len(set(topics))} -> {len(set(merged_topics))}")
            
            # 使用相同的UMAP降维配置
            reduced_embeddings_merged = UMAP(
                n_neighbors=10, 
                n_components=2, 
                min_dist=0.0, 
                metric='cosine'
            ).fit_transform(embeddings)
            
            # 创建简化的散点图
            df = pd.DataFrame({
                'x': reduced_embeddings_merged[:, 0],
                'y': reduced_embeddings_merged[:, 1],
                'topic': [merged_topic2label.get(t, f"主题{t}") for t in merged_topics]
            })
            
            fig = px.scatter(
                df, x='x', y='y', color='topic',
                title='合并主题后的文档分布图',
                labels={'x': 'UMAP 1', 'y': 'UMAP 2'}
            )
            fig.update_layout(
                width=1200, height=800,
                font=dict(size=12)
            )
            
            fig.write_html("results/5_5/8_merged_document_distribution.html")
            try:
                fig.write_image("results/5_5/8_merged_document_distribution.png", width=1200, height=800)
                print('✓ 8. 合并主题后的文档分布图片已保存')
            except Exception as e:
                print(f'⚠ 8. 合并主题后的文档分布图片导出失败: {e}')
        else:
            print('⚠ 主题未合并，跳过合并主题后的文档分布')
    except Exception as e:
        print(f'⚠ 合并主题后的文档分布生成失败: {e}')
    
    print("=== 所有可视化生成完成 ===")
    print(f'结果保存在: results/5_5/')
    print('可视化顺序与main.ipynb对齐:')
    print('  1. 主题关键词条形图 (Cell 12)')
    print('  2. 主题可视化 (Cell 13)')
    print('  3. 文档分布 (Cell 14)')
    print('  4. 主题层次聚类 (Cell 16)')
    print('  5. 主题相似性热力图 (BERTopic标准)')
    print('  6. 动态主题模型时间分布 (Cell 23)')
    print('  7. 合并主题后的主题可视化 (Cell 19)')
    print('  8. 合并主题后的文档分布 (Cell 20)')

if __name__ == "__main__":
    main() 