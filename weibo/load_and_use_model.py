#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加载已训练的BERTopic模型并进行预测
"""

import numpy as np
import pandas as pd
from bertopic import BERTopic
import jieba
import os
import sys
from datetime import datetime

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

def load_trained_model():
    """加载已训练的BERTopic模型"""
    print("=== 加载已训练的模型 ===")
    
    model_path = 'data/bertopic_model'
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在: {model_path}")
        print("请先运行 5_main_weibo.py 训练模型")
        return None
    
    try:
        topic_model = BERTopic.load(model_path)
        print(f"✓ 成功加载模型: {model_path}")
        print(f"✓ 模型主题数量: {len(topic_model.get_topics())}")
        return topic_model
    except Exception as e:
        print(f"✗ 加载模型失败: {e}")
        return None

def load_training_data():
    """加载训练数据信息"""
    print("\n=== 加载训练数据信息 ===")
    
    # 加载主题信息
    topic_info_path = 'data/topic_info.csv'
    if os.path.exists(topic_info_path):
        topic_info = pd.read_csv(topic_info_path)
        print(f"✓ 加载主题信息: {len(topic_info)} 个主题")
        return topic_info
    else:
        print(f"警告：主题信息文件不存在: {topic_info_path}")
        return None

def predict_new_texts(topic_model, new_texts):
    """对新文本进行主题预测"""
    print(f"\n=== 预测新文本主题 ===")
    print(f"新文本数量: {len(new_texts)}")
    
    try:
        # 预测主题
        topics, probs = topic_model.transform(new_texts)
        
        # 处理概率
        if probs is not None and probs.ndim > 1:
            topic_probabilities = np.max(probs, axis=1)
        else:
            topic_probabilities = probs if probs is not None else [1.0] * len(new_texts)
        
        # 创建结果DataFrame
        results = pd.DataFrame({
            'text': new_texts,
            'predicted_topic': topics,
            'topic_probability': topic_probabilities
        })
        
        # 添加主题标签
        topic_info = topic_model.get_topic_info()
        topic_label_dict = dict(zip(topic_info['Topic'], topic_info['Name']))
        results['topic_label'] = results['predicted_topic'].map(topic_label_dict)
        
        print("✓ 预测完成！")
        return results
        
    except Exception as e:
        print(f"✗ 预测失败: {e}")
        return None

def analyze_predictions(results, topic_model):
    """分析预测结果"""
    print("\n=== 预测结果分析 ===")
    
    if results is None:
        return
    
    # 统计主题分布
    topic_counts = results['predicted_topic'].value_counts().sort_index()
    print("预测主题分布:")
    for topic_id, count in topic_counts.head(10).items():
        if topic_id != -1:
            topic_words = topic_model.get_topic(topic_id)[:5]
            words_str = ', '.join([word for word, _ in topic_words])
            print(f"主题 {topic_id}: {count} 个文档 - {words_str}")
        else:
            print(f"异常值主题: {count} 个文档")
    
    # 显示详细结果
    print(f"\n详细预测结果:")
    for i, row in results.head(10).iterrows():
        print(f"文本{i+1}: {row['text'][:50]}...")
        print(f"预测主题: {row['predicted_topic']} ({row['topic_label']})")
        print(f"主题概率: {row['topic_probability']:.3f}")
        print()

def save_predictions(results, filename='new_texts_predictions.csv'):
    """保存预测结果"""
    if results is not None:
        results.to_csv(f'data/{filename}', index=False, encoding='utf-8')
        print(f"预测结果已保存到: data/{filename}")

def main():
    """主函数"""
    print("=" * 60)
    print("BERTopic模型加载和预测工具")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 加载模型
    topic_model = load_trained_model()
    if topic_model is None:
        return
    
    # 加载训练数据信息
    topic_info = load_training_data()
    
    # 示例新文本（你可以替换为自己的文本）
    new_texts = [
        "今天天气很好，我想去公园散步",
        "这家餐厅的菜很好吃，推荐大家来试试",
        "电影很精彩，演员表演很到位",
        "学习编程需要耐心和坚持",
        "旅游是放松心情的好方式",
        "健康饮食对身体很重要",
        "运动可以增强体质",
        "读书能开阔视野增长知识",
        "音乐能陶冶情操",
        "科技发展日新月异"
    ]
    
    # 预测新文本
    results = predict_new_texts(topic_model, new_texts)
    
    # 分析预测结果
    analyze_predictions(results, topic_model)
    
    # 保存预测结果
    save_predictions(results)
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("模型加载和预测完成！")
    print("=" * 60)
    
    print(f"\n使用说明:")
    print(f"1. 模型文件: data/bertopic_model")
    print(f"2. 主题信息: data/topic_info.csv")
    print(f"3. 预测结果: data/new_texts_predictions.csv")
    print(f"\n要预测自己的文本，请修改脚本中的 new_texts 列表")

if __name__ == "__main__":
    # 设置日志文件
    log_file = "data/log/load_model_log.txt"
    
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
        print(f"模型加载日志已保存到: {log_file}") 