#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据预处理脚本
按照论文方案处理微博数据
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from logging_config import get_data_processing_logger

# 获取日志记录器
logger = get_data_processing_logger()

def create_directories():
    """创建必要的目录"""
    directories = [
        'data/clean',
        'result',
        'result/sentiment',
        'result/behavior'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"创建目录: {directory}")

def load_and_clean_data():
    """加载和清洗数据"""
    logger.info("=== 数据预处理 ===")
    
    # 加载原始数据
    try:
        df = pd.read_excel('data/data_new.xlsx')
        logger.info(f"原始数据形状: {df.shape}")
        logger.info(f"列名: {list(df.columns)}")
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        raise
    
    # 数据清洗
    df_clean = df.copy()
    
    # 删除文本为空的记录
    initial_count = len(df_clean)
    df_clean = df_clean.dropna(subset=['txt'])
    df_clean = df_clean[df_clean['txt'].str.len() > 0]
    removed_count = initial_count - len(df_clean)
    logger.info(f"删除空文本记录: {removed_count} 条")
    
    # 处理时间字段
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_clean['year'] = df_clean['date'].dt.year
    df_clean['month'] = df_clean['date'].dt.month
    df_clean['day'] = df_clean['date'].dt.day
    
    # 添加文本长度字段
    df_clean['txt_length'] = df_clean['txt'].str.len()
    
    # 处理用户类型字段
    df_clean['user_type'] = df_clean['type'].fillna('个人')
    df_clean['user_type'] = df_clean['user_type'].str.strip()
    
    # 统一用户类型分类
    df_clean.loc[df_clean['user_type'].str.contains('官方', na=False), 'user_type'] = '官方组织'
    df_clean.loc[df_clean['user_type'].str.contains('个人', na=False), 'user_type'] = '个人'
    
    logger.info(f"清洗后数据形状: {df_clean.shape}")
    logger.info(f"时间范围: {df_clean['date'].min()} 至 {df_clean['date'].max()}")
    logger.info(f"用户类型分布:\n{df_clean['user_type'].value_counts()}")
    
    # 保存清洗后的数据
    try:
        df_clean.to_csv('data/clean/processed_data.csv', index=False, encoding='utf-8')
        logger.info("清洗后数据已保存到 data/clean/processed_data.csv")
    except Exception as e:
        logger.error(f"保存数据失败: {e}")
        raise
    
    return df_clean

def analyze_data_statistics(df):
    """分析数据统计信息"""
    logger.info("=== 数据统计分析 ===")
    
    # 基本统计
    logger.info(f"总记录数: {len(df)}")
    logger.info(f"平均文本长度: {df['txt_length'].mean():.2f}")
    logger.info(f"文本长度中位数: {df['txt_length'].median():.2f}")
    logger.info(f"最长文本: {df['txt_length'].max()} 字符")
    
    # 互动数据统计
    logger.info("互动数据统计:")
    for col in ['repost', 'comment', 'like']:
        if col in df.columns:
            logger.info(f"{col}: 平均 {df[col].mean():.2f}, 最大 {df[col].max()}")
    
    # 时间分布
    logger.info(f"时间分布:")
    logger.info(f"最早日期: {df['date'].min()}")
    logger.info(f"最晚日期: {df['date'].max()}")
    logger.info(f"数据跨度: {(df['date'].max() - df['date'].min()).days} 天")
    
    # 保存统计报告
    stats_report = f"""
数据统计报告
============
总记录数: {len(df)}
时间范围: {df['date'].min()} 至 {df['date'].max()}
数据跨度: {(df['date'].max() - df['date'].min()).days} 天

文本长度统计:
- 平均长度: {df['txt_length'].mean():.2f} 字符
- 中位数长度: {df['txt_length'].median():.2f} 字符
- 最长文本: {df['txt_length'].max()} 字符
- 最短文本: {df['txt_length'].min()} 字符

用户类型分布:
{df['user_type'].value_counts().to_string()}

互动数据统计:
- 平均转发数: {df['repost'].mean():.2f}
- 平均评论数: {df['comment'].mean():.2f}
- 平均点赞数: {df['like'].mean():.2f}
"""
    
    try:
        with open('result/data_statistics.txt', 'w', encoding='utf-8') as f:
            f.write(stats_report)
        logger.info("统计报告已保存到 result/data_statistics.txt")
    except Exception as e:
        logger.error(f"保存统计报告失败: {e}")

def main():
    """主函数"""
    logger.info("开始数据预处理...")
    
    try:
        # 创建目录
        create_directories()
        
        # 加载和清洗数据
        df_clean = load_and_clean_data()
        
        # 分析数据统计
        analyze_data_statistics(df_clean)
        
        logger.info("数据预处理完成！")
        return df_clean
        
    except Exception as e:
        logger.error(f"数据预处理过程中发生错误: {e}")
        raise

if __name__ == "__main__":
    main() 