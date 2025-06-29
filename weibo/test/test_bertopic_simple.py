#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的BERTopic测试脚本
"""

import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import jieba
import time

def jieba_tokenizer(text):
    """jieba分词器"""
    return jieba.lcut(text)

def test_bertopic():
    """测试BERTopic基本功能"""
    print("=== BERTopic简单测试 ===")
    
    # 扩充测试数据到20条
    test_texts = [
        "今天天气很好，我想去公园散步",
        "这家餐厅的菜很好吃，推荐大家来试试",
        "电影很精彩，演员表演很到位",
        "学习编程需要耐心和坚持",
        "旅游是放松心情的好方式",
        "健康饮食对身体很重要",
        "运动可以增强体质",
        "读书能开阔视野增长知识",
        "音乐能陶冶情操",
        "科技发展日新月异",
        "昨晚的足球比赛非常精彩",
        "我喜欢喝咖啡，提神醒脑",
        "春天百花盛开，景色宜人",
        "手机更新了新系统，体验更流畅",
        "朋友聚会总是让人开心",
        "下雨天适合在家看书",
        "人工智能正在改变世界",
        "养宠物可以带来很多乐趣",
        "节假日高速公路很拥堵",
        "环保意识越来越被重视"
    ]
    
    print(f"测试文本数量: {len(test_texts)}")
    
    # 创建简单的嵌入向量（随机生成用于测试）
    np.random.seed(42)
    test_embeddings = np.random.rand(len(test_texts), 384)
    print(f"嵌入向量形状: {test_embeddings.shape}")
    
    # 创建BERTopic模型
    print("\n创建BERTopic模型...")
    vectorizer = CountVectorizer(
        analyzer='word',
        tokenizer=jieba_tokenizer,
        min_df=1,
        max_df=1.0,
        stop_words=None
    )
    
    topic_model = BERTopic(
        min_topic_size=2,
        vectorizer_model=vectorizer,
        verbose=True
    )
    
    # 训练模型
    print("\n开始训练...")
    start_time = time.time()
    
    try:
        # topics, probs = topic_model.fit_transform(test_texts, test_embeddings)
        topics, probs = topic_model.fit_transform(test_texts)
        
        training_time = time.time() - start_time
        print(f"✓ 训练成功！用时: {training_time:.2f}秒")
        print(f"✓ 发现主题数量: {len(topic_model.get_topics())}")
        
        # 显示结果
        print("\n=== 训练结果 ===")
        for i, (text, topic) in enumerate(zip(test_texts, topics)):
            print(f"文本{i+1}: {text}")
            print(f"主题: {topic}")
            print()
        
        # 显示主题信息
        topic_info = topic_model.get_topic_info()
        print("主题信息:")
        print(topic_info)
        
        return True
        
    except Exception as e:
        print(f"✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_bertopic()
    if success:
        print("\n✓ BERTopic测试成功！")
    else:
        print("\n✗ BERTopic测试失败！") 