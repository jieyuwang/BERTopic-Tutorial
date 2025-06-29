#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速创建原始文本文件，避免重新生成嵌入向量
"""

import os
import sys
from datetime import datetime

def create_original_texts():
    """创建原始文本文件"""
    print("=== 快速创建原始文本文件 ===")
    
    # 检查嵌入向量数量
    try:
        import numpy as np
        embeddings = np.load('embedding/emb.npy')
        embedding_count = embeddings.shape[0]
        print(f"嵌入向量数量: {embedding_count:,}")
    except Exception as e:
        print(f"无法读取嵌入向量: {e}")
        return False
    
    # 读取原始文本
    try:
        with open('data/文本.txt', 'r', encoding='utf-8') as f:
            original_texts = [line.strip() for line in f if line.strip()]
        print(f"原始文本数量: {len(original_texts):,}")
    except Exception as e:
        print(f"无法读取原始文本: {e}")
        return False
    
    # 确保数量匹配
    if len(original_texts) < embedding_count:
        print(f"⚠️  警告：原始文本数量({len(original_texts)})少于嵌入向量数量({embedding_count})")
        print("将使用所有可用的原始文本")
        texts_to_save = original_texts
    else:
        texts_to_save = original_texts[:embedding_count]
    
    # 保存原始文本
    try:
        with open('embedding/original_texts.txt', 'w', encoding='utf-8') as f:
            for text in texts_to_save:
                f.write(text + '\n')
        
        print(f"✓ 成功保存 {len(texts_to_save):,} 条原始文本到 embedding/original_texts.txt")
        return True
        
    except Exception as e:
        print(f"保存失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("快速创建原始文本文件")
    print("=" * 50)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    success = create_original_texts()
    
    if success:
        print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)
        print("原始文本文件创建完成！")
        print("=" * 50)
        print("\n现在可以直接运行: python 5_main_weibo.py")
    else:
        print("\n创建失败，请检查文件路径")

if __name__ == "__main__":
    main() 