#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
词汇分析脚本 - 分析Excel数据中的高频词汇，为优化用户自定义词典提供建议
"""

import pandas as pd
import jieba
import re
from collections import Counter
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import warnings
import sys
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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

def load_data():
    """加载Excel数据"""
    try:
        # 尝试读取Excel文件
        df = pd.read_excel('data/data_new.xlsx')
        print(f"成功加载数据，共 {len(df)} 行")
        print(f"列名: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

def preprocess_text(text):
    """文本预处理"""
    if pd.isna(text):
        return ""
    
    # 转换为字符串
    text = str(text)
    
    # 去除特殊字符，保留中文、英文、数字
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
    
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_stopwords():
    """加载停用词"""
    try:
        with open('分词/stopwords.txt', 'r', encoding='utf-8') as f:
            stopwords = set([line.strip() for line in f if line.strip()])
        print(f"加载停用词 {len(stopwords)} 个")
        return stopwords
    except:
        print("未找到停用词文件，使用空集合")
        return set()

def load_userdict():
    """加载用户自定义词典"""
    try:
        with open('分词/userdict.txt', 'r', encoding='utf-8') as f:
            userdict = set([line.strip() for line in f if line.strip()])
        print(f"加载用户自定义词典 {len(userdict)} 个")
        return userdict
    except:
        print("未找到用户自定义词典文件")
        return set()

def analyze_vocabulary(df, text_column='txt'):
    """分析词汇频率"""
    print("\n=== 开始词汇分析 ===")
    
    # 加载停用词和用户词典
    stopwords = load_stopwords()
    userdict = load_userdict()
    
    # 添加用户词典到jieba
    for word in userdict:
        jieba.add_word(word)
    
    # 文本预处理和分词
    all_words = []
    all_texts = []
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"处理进度: {idx}/{len(df)}")
        
        text = preprocess_text(row.get(text_column, ''))
        if not text:
            continue
            
        all_texts.append(text)
        
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
        
        all_words.extend(filtered_words)
    
    # 统计词频
    word_freq = Counter(all_words)
    
    print(f"\n总词汇数: {len(all_words):,}")
    print(f"唯一词汇数: {len(word_freq):,}")
    
    return word_freq, all_texts, all_words

def suggest_userdict_words(word_freq, min_freq=10, min_length=2, max_length=10):
    """建议用户自定义词典词汇"""
    print(f"\n=== 用户自定义词典建议 ===")
    print(f"筛选条件: 频率 >= {min_freq}, 长度 {min_length}-{max_length}")
    
    # 筛选候选词汇
    candidates = []
    for word, freq in word_freq.items():
        if (freq >= min_freq and 
            min_length <= len(word) <= max_length and
            not re.match(r'^[a-zA-Z]+$', word)):  # 排除纯英文
            candidates.append((word, freq))
    
    # 按频率排序
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n候选词汇数量: {len(candidates):,}")
    
    # 显示Top 50
    print("\nTop 50 高频词汇建议:")
    print("-" * 50)
    for i, (word, freq) in enumerate(candidates[:50]):
        print(f"{i+1:2d}. {word:15s} (频率: {freq:4d})")
    
    return candidates

def analyze_word_patterns(word_freq):
    """分析词汇模式"""
    print(f"\n=== 词汇模式分析 ===")
    
    # 按长度分组
    length_groups = {}
    for word, freq in word_freq.items():
        length = len(word)
        if length not in length_groups:
            length_groups[length] = []
        length_groups[length].append((word, freq))
    
    print("\n按词汇长度分布:")
    for length in sorted(length_groups.keys()):
        words = length_groups[length]
        total_freq = sum(freq for _, freq in words)
        print(f"长度 {length}: {len(words):,} 个词, 总频率 {total_freq:,}")
        
        # 显示该长度下的Top 10
        top_words = sorted(words, key=lambda x: x[1], reverse=True)[:10]
        print(f"  Top 10: {[word for word, _ in top_words]}")

def create_wordcloud(word_freq, output_file='data/wordcloud.png'):
    """生成词云图"""
    print(f"\n=== 生成词云图 ===")
    
    # 生成词云
    wordcloud = WordCloud(
        font_path='/System/Library/Fonts/PingFang.ttc',  # macOS中文字体
        width=1200, 
        height=800,
        background_color='white',
        max_words=200,
        colormap='viridis'
    )
    
    # 准备词频数据
    word_freq_dict = dict(word_freq.most_common(100))
    
    # 生成词云
    wordcloud.generate_from_frequencies(word_freq_dict)
    
    # 保存图片
    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('词汇频率词云图', fontsize=20, pad=20)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"词云图已保存到: {output_file}")

def analyze_topic_keywords(word_freq, all_texts):
    """分析主题相关关键词"""
    print(f"\n=== 主题关键词分析 ===")
    
    # 使用TF-IDF提取关键词
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # 准备文本数据
    texts = [' '.join(jieba.lcut(text)) for text in all_texts if text.strip()]
    
    # TF-IDF向量化
    tfidf = TfidfVectorizer(
        max_features=1000,
        min_df=5,
        max_df=0.8,
        stop_words=None  # 我们已经在分词时处理了停用词
    )
    
    try:
        tfidf_matrix = tfidf.fit_transform(texts)
        feature_names = tfidf.get_feature_names_out()
        
        # 计算平均TF-IDF分数
        avg_tfidf = np.mean(tfidf_matrix.toarray(), axis=0)
        
        # 获取Top关键词
        top_indices = np.argsort(avg_tfidf)[-50:][::-1]
        
        print("\nTF-IDF Top 50 关键词:")
        print("-" * 50)
        for i, idx in enumerate(top_indices):
            word = feature_names[idx]
            score = avg_tfidf[idx]
            freq = word_freq.get(word, 0)
            print(f"{i+1:2d}. {word:15s} (TF-IDF: {score:.4f}, 频率: {freq})")
            
    except Exception as e:
        print(f"TF-IDF分析失败: {e}")

def generate_updated_userdict(candidates, output_file='分词/userdict_updated.txt'):
    """生成更新后的用户自定义词典"""
    print(f"\n=== 生成更新后的用户自定义词典 ===")
    
    # 读取原有词典
    try:
        with open('分词/userdict.txt', 'r', encoding='utf-8') as f:
            original_words = set([line.strip() for line in f if line.strip()])
    except:
        original_words = set()
    
    # 选择Top 100个候选词汇
    new_words = [word for word, _ in candidates[:100]]
    
    # 合并词汇
    all_words = list(original_words) + new_words
    all_words = list(set(all_words))  # 去重
    
    # 保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for word in sorted(all_words):
            f.write(word + '\n')
    
    print(f"更新后的词典已保存到: {output_file}")
    print(f"原有词汇: {len(original_words)} 个")
    print(f"新增词汇: {len(new_words)} 个")
    print(f"总词汇: {len(all_words)} 个")

def main():
    """主函数"""
    print("=" * 60)
    print("微博数据词汇分析工具")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 加载数据
    df = load_data()
    if df is None:
        return
    
    # 分析词汇
    word_freq, all_texts, all_words = analyze_vocabulary(df)
    
    # 建议用户自定义词典词汇
    candidates = suggest_userdict_words(word_freq, min_freq=5)
    
    # 分析词汇模式
    analyze_word_patterns(word_freq)
    
    # 分析主题关键词
    analyze_topic_keywords(word_freq, all_texts)
    
    # 生成词云图
    create_wordcloud(word_freq)
    
    # 生成更新后的用户自定义词典
    generate_updated_userdict(candidates)
    
    # 生成详细统计报告
    print(f"\n" + "=" * 40)
    print("词汇分析统计报告")
    print("=" * 40)
    print(f"总文本数: {len(all_texts):,}")
    print(f"总词汇数: {len(all_words):,}")
    print(f"唯一词汇数: {len(word_freq):,}")
    print(f"候选词典词汇数: {len(candidates):,}")
    
    # 词汇长度分布统计
    length_dist = {}
    for word in word_freq.keys():
        length = len(word)
        length_dist[length] = length_dist.get(length, 0) + 1
    
    print(f"\n词汇长度分布:")
    for length in sorted(length_dist.keys()):
        count = length_dist[length]
        percentage = (count / len(word_freq)) * 100
        print(f"  长度{length}: {count:,} 个词 ({percentage:.1f}%)")
    
    # 高频词汇统计
    top_words = word_freq.most_common(20)
    print(f"\nTop 20 高频词汇:")
    for i, (word, freq) in enumerate(top_words):
        percentage = (freq / len(all_words)) * 100
        print(f"  {i+1:2d}. {word:15s}: {freq:,} 次 ({percentage:.2f}%)")
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("词汇分析完成！")
    print("=" * 60)
    
    print(f"\n建议:")
    print("1. 查看 'data/wordcloud.png' 了解词汇分布")
    print("2. 查看 '分词/userdict_updated.txt' 获取更新后的词典")
    print("3. 根据业务需求调整词典内容")

if __name__ == "__main__":
    # 设置日志文件
    log_file = "data/log/02_analyze_vocabulary_log.txt"
    
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
        print(f"词汇分析日志已保存到: {log_file}") 