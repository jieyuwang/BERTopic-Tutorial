#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
嵌入向量生成速度测试脚本
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import time
import torch

def test_embedding_speed():
    """测试嵌入向量生成速度"""
    print("=" * 50)
    print("嵌入向量生成速度测试")
    print("=" * 50)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 测试模型
    model_name = 'all-MiniLM-L6-v2'
    print(f"测试模型: {model_name}")
    
    # 加载模型
    print("正在加载模型...")
    start_time = time.time()
    model = SentenceTransformer(model_name)
    if device.type == 'cuda':
        model = model.to(device)
    load_time = time.time() - start_time
    print(f"模型加载用时: {load_time:.2f}秒")
    
    # 准备测试数据
    test_texts = [
        "这是一个测试文本",
        "今天天气很好",
        "我喜欢机器学习",
        "BERTopic是一个很好的主题建模工具",
        "中文文本处理很有趣"
    ] * 100  # 重复100次，得到500个文本
    
    print(f"测试文本数量: {len(test_texts)}")
    
    # 测试不同批次大小
    batch_sizes = [1, 8, 16, 32, 64, 128]
    
    for batch_size in batch_sizes:
        print(f"\n测试批次大小: {batch_size}")
        
        # 预热
        if batch_size <= 8:
            warmup_texts = test_texts[:batch_size]
            model.encode(warmup_texts, show_progress_bar=False)
        
        # 实际测试
        start_time = time.time()
        embeddings = model.encode(test_texts, batch_size=batch_size, show_progress_bar=False)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_text = total_time / len(test_texts)
        texts_per_second = len(test_texts) / total_time
        
        print(f"  总用时: {total_time:.2f}秒")
        print(f"  平均每文本: {avg_time_per_text*1000:.2f}毫秒")
        print(f"  处理速度: {texts_per_second:.1f} 文本/秒")
        print(f"  向量形状: {embeddings.shape}")
    
    # 估算完整数据集的处理时间
    print(f"\n" + "="*50)
    print("完整数据集处理时间估算")
    print("="*50)
    
    full_dataset_size = 86816  # 您的数据集大小
    best_speed = 50  # 假设最佳速度为50文本/秒（保守估计）
    
    estimated_time_minutes = full_dataset_size / best_speed / 60
    estimated_time_hours = estimated_time_minutes / 60
    
    print(f"数据集大小: {full_dataset_size:,} 文本")
    print(f"预计处理时间: {estimated_time_minutes:.1f} 分钟 ({estimated_time_hours:.1f} 小时)")
    
    # 建议
    print(f"\n建议:")
    print(f"1. 如果处理时间过长，考虑减小批次大小")
    print(f"2. 如果有GPU，确保使用GPU加速")
    print(f"3. 可以考虑使用更轻量级的模型")
    print(f"4. 分批处理，每批处理部分数据")

if __name__ == "__main__":
    test_embedding_speed() 