#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
微博数据预处理脚本
将Excel数据转换为BERTopic处理所需的格式
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
import os
import sys
from io import StringIO

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

def clean_text(text):
    """清理文本内容"""
    if pd.isna(text) or text == '':
        return ''
    
    # 转换为字符串
    text = str(text)
    
    # 移除URL
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # 移除@用户
    text = re.sub(r'@[\w\-]+', '', text)
    
    # 移除#话题#
    text = re.sub(r'#.*?#', '', text)
    
    # 移除特殊字符，保留中文、英文、数字和基本标点
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？；：""''（）【】]', '', text)
    
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_date_info(date_str):
    """提取日期信息"""
    try:
        if pd.isna(date_str):
            return '2023-01-01'
        
        # 如果是datetime对象
        if isinstance(date_str, datetime):
            return date_str.strftime('%Y-%m-%d')
        
        # 如果是字符串
        date_str = str(date_str)
        
        # 尝试解析不同格式的日期
        for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%Y年%m月%d日']:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime('%Y-%m-%d')
            except:
                continue
        
        # 如果都解析失败，返回默认日期
        return '2023-01-01'
    except:
        return '2023-01-01'

def preprocess_weibo_data(input_file, output_dir):
    """预处理微博数据"""
    print("=" * 60)
    print("微博数据预处理开始")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("开始读取Excel文件...")
    
    # 读取Excel文件
    df = pd.read_excel(input_file)
    print(f"原始数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # 清理数据
    print("\n开始清理数据...")
    
    # 移除空行和无效数据
    df = df.dropna(subset=['txt'])
    df = df[df['txt'].str.strip() != '']
    
    # 清理文本内容
    df['clean_txt'] = df['txt'].apply(clean_text)
    
    # 移除清理后为空的文本
    df = df[df['clean_txt'] != '']
    
    # 提取日期信息
    df['date_clean'] = df['date'].apply(extract_date_info)
    
    # 提取年份
    df['year'] = df['date_clean'].apply(lambda x: x.split('-')[0] if '-' in str(x) else '2023')
    
    print(f"清理后数据形状: {df.shape}")
    
    # 保存清理后的数据
    print("\n保存清理后的数据...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存文本数据
    with open(os.path.join(output_dir, '文本.txt'), 'w', encoding='utf-8') as f:
        for text in df['clean_txt']:
            f.write(text + '\n')
    
    # 保存时间数据
    with open(os.path.join(output_dir, '时间.txt'), 'w', encoding='utf-8') as f:
        for year in df['year']:
            f.write(str(year) + '\n')
    
    # 保存完整数据（用于后续分析）
    df_clean = df[['user', 'clean_txt', 'date_clean', 'year', 'repost', 'comment', 'like', 'type']].copy()
    df_clean.to_csv(os.path.join(output_dir, 'weibo_clean_data.csv'), index=False, encoding='utf-8')
    
    # 生成详细统计信息
    stats = {
        '总数据量': len(df),
        '用户数量': df['user'].nunique(),
        '时间范围': f"{df['date_clean'].min()} 到 {df['date_clean'].max()}",
        '平均转发数': round(df['repost'].mean(), 2),
        '平均评论数': round(df['comment'].mean(), 2),
        '平均点赞数': round(df['like'].mean(), 2),
        '用户类型分布': df['type'].value_counts().to_dict()
    }
    
    print("\n" + "=" * 40)
    print("数据统计信息")
    print("=" * 40)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 用户类型分布详细信息
    print("\n用户类型详细分布:")
    type_counts = df['type'].value_counts()
    for type_name, count in type_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {type_name}: {count} ({percentage:.1f}%)")
    
    # 时间分布信息
    print("\n时间分布信息:")
    year_counts = df['year'].value_counts().sort_index()
    for year, count in year_counts.head(10).items():
        percentage = (count / len(df)) * 100
        print(f"  {year}年: {count} ({percentage:.1f}%)")
    
    print(f"\n" + "=" * 40)
    print("文件保存信息")
    print("=" * 40)
    print(f"数据已保存到: {output_dir}")
    print(f"- 文本.txt: {len(df)} 条文本")
    print(f"- 时间.txt: {len(df)} 条时间记录")
    print(f"- weibo_clean_data.csv: 完整清理数据")
    
    # 文件大小信息
    text_file_size = os.path.getsize(os.path.join(output_dir, '文本.txt'))
    time_file_size = os.path.getsize(os.path.join(output_dir, '时间.txt'))
    csv_file_size = os.path.getsize(os.path.join(output_dir, 'weibo_clean_data.csv'))
    
    print(f"\n文件大小信息:")
    print(f"- 文本.txt: {text_file_size:,} bytes ({text_file_size/1024:.1f} KB)")
    print(f"- 时间.txt: {time_file_size:,} bytes ({time_file_size/1024:.1f} KB)")
    print(f"- weibo_clean_data.csv: {csv_file_size:,} bytes ({csv_file_size/1024:.1f} KB)")
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("数据预处理完成！")
    print("=" * 60)
    
    return df_clean

if __name__ == "__main__":
    # 设置路径
    input_file = "data/data_new.xlsx"
    output_dir = "data"
    log_file = "data/log/01_data_preprocessing_log.txt"
    
    # 确保日志目录存在
    os.makedirs('data/log', exist_ok=True)
    
    # 设置日志记录
    logger = Logger(log_file)
    sys.stdout = logger
    
    try:
        # 执行预处理
        df_clean = preprocess_weibo_data(input_file, output_dir)
    finally:
        # 恢复标准输出并关闭日志文件
        sys.stdout = sys.__stdout__
        logger.close()
        print(f"数据预处理日志已保存到: {log_file}") 