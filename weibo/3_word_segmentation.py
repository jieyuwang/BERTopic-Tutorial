#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分词脚本 - 对预处理后的文本进行中文分词
"""

import jieba
import pandas as pd
import re
import sys
import os
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

def load_stopwords():
    """加载停用词"""
    try:
        with open('分词/stopwords.txt', 'r', encoding='utf-8') as f:
            stopwords = set([line.strip() for line in f if line.strip()])
        print(f"加载停用词 {len(stopwords)} 个")
        return stopwords
    except Exception as e:
        print(f"加载停用词失败: {e}")
        return set()

def load_userdict():
    """加载用户自定义词典"""
    try:
        with open('分词/userdict.txt', 'r', encoding='utf-8') as f:
            userdict = set([line.strip() for line in f if line.strip()])
        print(f"加载用户自定义词典 {len(userdict)} 个")
        return userdict
    except Exception as e:
        print(f"加载用户自定义词典失败: {e}")
        return set()

def segment_text(text, stopwords, userdict):
    """对文本进行分词"""
    if pd.isna(text) or not text.strip():
        return ""
    
    # 添加用户词典词汇到jieba
    for word in userdict:
        jieba.add_word(word)
    
    # 分词
    words = jieba.lcut(text)
    
    # 过滤停用词和短词
    filtered_words = []
    for word in words:
        word = word.strip()
        if (len(word) >= 2 and 
            word not in stopwords and 
            not word.isdigit() and
            not re.match(r'^[a-zA-Z]+$', word)):
            filtered_words.append(word)
    
    return ' '.join(filtered_words)

def main():
    """主函数"""
    print("=" * 60)
    print("微博数据分词处理工具")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 加载数据
    try:
        df = pd.read_csv('data/weibo_clean_data.csv')
        print(f"成功加载清洗后的数据，共 {len(df):,} 行")
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    # 加载停用词和用户词典
    stopwords = load_stopwords()
    userdict = load_userdict()
    
    # 分词处理
    print("\n=== 开始分词处理 ===")
    segmented_texts = []
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"处理进度: {idx:,}/{len(df):,}")
        
        text = row.get('clean_txt', '')
        segmented_text = segment_text(text, stopwords, userdict)
        segmented_texts.append(segmented_text)
    
    # 保存分词结果
    print("\n=== 保存分词结果 ===")
    
    # 保存到CSV文件
    df['segmented_text'] = segmented_texts
    df.to_csv('data/weibo_segmented_data.csv', index=False, encoding='utf-8')
    print(f"分词结果已保存到: data/weibo_segmented_data.csv")
    
    # 保存分词文本到txt文件
    with open('data/切词.txt', 'w', encoding='utf-8') as f:
        for text in segmented_texts:
            if text.strip():
                f.write(text + '\n')
    print(f"分词文本已保存到: data/切词.txt")
    
    # 统计信息
    print(f"\n=== 分词统计信息 ===")
    total_segments = sum(len(text.split()) for text in segmented_texts if text.strip())
    non_empty_segments = sum(1 for text in segmented_texts if text.strip())
    
    print(f"总文本数: {len(df):,}")
    print(f"有效分词文本数: {non_empty_segments:,}")
    print(f"总词汇数: {total_segments:,}")
    
    # 检查是否有有效的分词文本
    if non_empty_segments > 0:
        avg_segments = total_segments / non_empty_segments
        print(f"平均每文本词汇数: {avg_segments:.1f}")
    else:
        print("警告：没有生成有效的分词文本！")
        print("可能的原因：")
        print("1. 原始文本为空或格式不正确")
        print("2. 停用词过滤过于严格")
        print("3. 用户词典配置问题")
    
    # 显示一些分词示例
    print(f"\n=== 分词示例 ===")
    example_count = 0
    for i, text in enumerate(segmented_texts):
        if text.strip() and example_count < 5:
            print(f"示例 {example_count+1}: {text[:100]}...")
            example_count += 1
    
    if example_count == 0:
        print("没有找到有效的分词示例")
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("分词处理完成！")
    print("=" * 60)

if __name__ == "__main__":
    # 设置日志文件
    log_file = "data/log/03_word_segmentation_log.txt"
    
    # 设置日志记录
    logger = Logger(log_file)
    sys.stdout = logger
    
    try:
        main()
    finally:
        # 恢复标准输出并关闭日志文件
        sys.stdout = sys.__stdout__
        logger.close()
        print(f"分词处理日志已保存到: {log_file}") 