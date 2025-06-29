#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试BERTopic优化效果
"""

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import jieba
import time
import os

# 设置并行处理
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'

def jieba_tokenizer(text):
    """jieba分词器"""
    return jieba.lcut(text)

def test_bertopic_optimized():
    """测试优化后的BERTopic"""
    print("=== 快速测试BERTopic优化效果 ===")
    
    # 加载小样本数据
    try:
        embeddings = np.load('../embedding/emb.npy')
        with open('../embedding/original_texts.txt', 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        # 使用前5000个样本进行快速测试
        test_size = 5000
        test_embeddings = embeddings[:test_size]
        test_texts = texts[:test_size]
        
        print(f"测试样本数量: {test_size}")
        print(f"嵌入向量形状: {test_embeddings.shape}")
        
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    # 创建优化的BERTopic模型
    print("\n创建优化的BERTopic模型...")
    vectorizer = CountVectorizer(
        analyzer='word',
        tokenizer=jieba_tokenizer,
        max_features=1000,  # 进一步减少词汇表
        min_df=2,
        max_df=0.95,
        stop_words=None,
        ngram_range=(1, 1)
    )
    
    topic_model = BERTopic(
        min_topic_size=20,
        top_n_words=15,
        vectorizer_model=vectorizer,
        verbose=True,
        calculate_probabilities=True
    )
    
    # 训练模型
    print("\n开始训练...")
    start_time = time.time()
    
    try:
        topics, probs = topic_model.fit_transform(test_texts, test_embeddings)
        
        training_time = time.time() - start_time
        print(f"✓ 训练成功！")
        print(f"✓ 训练用时: {training_time:.1f}秒")
        print(f"✓ 发现主题数量: {len(topic_model.get_topics())}")
        
        # 分析结果
        topic_info = topic_model.get_topic_info()
        print(f"\n主题信息:")
        print(topic_info.head(5))
        
        # 显示前几个主题的关键词
        print(f"\n前5个主题的关键词:")
        for i, row in topic_info.head(5).iterrows():
            topic_id = row['Topic']
            if topic_id != -1:
                topic_words = topic_model.get_topic(topic_id)[:8]
                words_str = ', '.join([word for word, _ in topic_words])
                print(f"主题 {topic_id}: {words_str}")
        
        # 统计异常值
        outlier_count = len([t for t in topics if t == -1])
        print(f"\n异常值统计:")
        print(f"  异常值数量: {outlier_count}")
        print(f"  异常值比例: {outlier_count/len(topics)*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_bertopic_optimized()
    if success:
        print("\n✓ 优化测试成功！")
        print("现在可以运行完整的 5_main_weibo.py")
    else:
        print("\n✗ 优化测试失败！") 