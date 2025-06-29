#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析原始微博数据，评估行为分类的合理性
专注于信息搜索行为和亲社会行为的分析
"""

import pandas as pd
import numpy as np
import jieba
from collections import Counter
import re
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
def setup_logger():
    """设置日志记录器"""
    os.makedirs('data/log', exist_ok=True)
    
    logger = logging.getLogger('focused_behavior_analysis')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = f'data/log/focused_behavior_analysis_{timestamp}.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()

def analyze_raw_data():
    """分析原始微博数据"""
    logger.info("=== 原始微博数据分析 ===")
    
    # 加载数据
    df = pd.read_excel('data/data_new.xlsx')
    logger.info(f"数据形状: {df.shape}")
    
    # 基本统计
    logger.info(f"\n=== 基本统计 ===")
    logger.info(f"总记录数: {len(df)}")
    logger.info(f"非空文本数: {df['txt'].notna().sum()}")
    logger.info(f"平均文本长度: {df['txt'].str.len().mean():.2f}")
    
    # 查看前10条微博
    logger.info(f"\n=== 前10条微博内容 ===")
    for i, row in df.head(10).iterrows():
        text = str(row['txt'])[:150]
        logger.info(f"{i+1}. {text}...")
    
    # 随机抽样分析
    logger.info(f"\n=== 随机抽样20条微博 ===")
    sample = df.sample(n=20, random_state=42)
    for i, row in sample.iterrows():
        text = str(row['txt'])[:150]
        logger.info(f"{i+1}. {text}...")
    
    # 关键词分析
    logger.info(f"\n=== 关键词分析 ===")
    all_text = ' '.join(df['txt'].dropna().astype(str))
    words = jieba.lcut(all_text)
    
    # 过滤停用词和短词
    stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也',
                  '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
    filtered_words = [word for word in words if len(word) > 1 and word not in stop_words]
    
    # 统计高频词
    word_counts = Counter(filtered_words)
    logger.info(f"高频词汇 (前20):")
    for word, count in word_counts.most_common(20):
        logger.info(f"  {word}: {count}")
    
    return df, word_counts

def analyze_behavior_keywords():
    """分析行为相关关键词"""
    logger.info(f"\n=== 行为关键词分析 ===")
    
    keywords_to_check = {
        '信息搜索相关': ['请问', '谁知道', '求问', '帮忙', '需要', '了解', '确认', '核实', '求证', '是否', '真假', '真的吗', '关注', '跟踪', '更新', '最新', '情况', '进展', '建议', '怎么办', '如何', '应该', '推荐', '意见'],
        '亲社会行为相关': ['求助', '求救', '救命', '紧急', '被困', '危险', '需要帮助', '帮助', '支援', '捐赠', '志愿者', '互助', '团结', '一起', '提供', '分享', '免费', '开放', '可用', '有需要', '加油', '支持', '鼓励', '安慰', '理解', '陪伴', '感谢', '感动', '温暖', '希望', '信心', '坚强']
    }
    
    for category, keywords in keywords_to_check.items():
        logger.info(f"\n{category}:")
        for keyword in keywords:
            logger.info(f"  {keyword}")

def evaluate_behavior_categories():
    """评估当前行为分类的合理性"""
    logger.info(f"\n=== 行为分类合理性评估 ===")
    
    # 当前的行为分类
    current_categories = {
        '基础设施中断行为': {
            '给排水': ['停水', '积水', '溢洪道', '管网堵塞', '排水不畅', '内涝'],
            '交通': ['路面塌陷', '道岔故障', '路基冲毁', '列车停运', '道路封闭', '交通中断'],
            '电力': ['供配电设施损坏', '备用电源失效', '区域断电', '停电', '电力中断'],
            '通讯': ['通信设施损坏', '信号中断', '防汛专用通信网络缺失', '网络中断']
        },
        '基础设施修复行为': {
            '给排水': ['水位监测', '积水抽排', '清淤', '封堵河道缺口', '应急供水'],
            '交通': ['路段排查', '道路封闭', '抢通作业', '复轨', '交通疏导', '道路畅通'],
            '电力': ['缆线加固', '设备巡检', '电缆抢修', '临时供电', '电力恢复'],
            '通讯': ['缆线加固', '设备巡检', '基站抢修', '建立应急通讯平台', '信号恢复']
        },
        '政府救助行为': {
            '转移': ['避险转移', '疏散', '撤离', '转移安置'],
            '救援': ['搜寻营救', '救援', '抢险', '救助'],
            '医疗': ['医疗救治', '医疗救助', '医疗支援'],
            '保障': ['生活保障', '物资保障', '应急保障'],
            '宣传': ['避险知识宣传', '安全宣传', '预警信息']
        },
        '公众行为': {
            '求助': ['求助', '求救', '请求帮助', '需要救援'],
            '互助': ['互助', '帮助', '支援', '捐赠', '志愿者'],
            '传谣': ['谣言', '虚假信息', '不实信息'],
            '生活': ['日常生活', '工作', '学习', '出行']
        }
    }
    
    logger.info("当前行为分类存在的问题:")
    logger.info("1. 基础设施中断/修复行为过于技术化，微博用户很少使用这些专业术语")
    logger.info("2. 政府救助行为分类合理，但关键词可能需要调整")
    logger.info("3. 公众行为分类过于简单，没有涵盖微博的主要表达方式")
    logger.info("4. 缺少对微博特有表达方式的分析（如转发、评论、@用户等）")
    
    return current_categories

def suggest_focused_categories():
    """建议聚焦的行为分类：信息搜索行为和亲社会行为"""
    logger.info(f"\n=== 聚焦的行为分类建议 ===")
    
    focused_categories = {
        '紧急求助行为': {
            '生命安全求助': ['救命', '被困', '危险', '紧急', '快', '急', '紧急情况', '生命危险', '生死', '危急'],
            '生活需求求助': ['求助', '需要帮助', '帮忙', '支持', '援助', '求援', '请求', '需要', '急需'],
            '信息需求求助': ['请问', '谁知道', '求问', '咨询', '了解', '求教', '问一下', '求助信息']
        },
        '信息搜索行为': {
            '主动询问': ['请问', '谁知道', '求问', '帮忙', '需要', '了解', '求教', '问一下', '求助', '咨询'],
            '信息求证': ['真的吗', '确认', '核实', '求证', '是否', '真假', '属实', '准确', '可靠', '可信', '官方'],
            '关注动态': ['关注', '跟踪', '更新', '最新', '情况', '进展', '状态', '消息', '通报', '公告'],
            '寻求建议': ['建议', '怎么办', '如何', '应该', '推荐', '意见', '指导', '帮助', '方案', '对策']
        },
        '互助支援行为': {
            '紧急救援': ['救援', '抢险', '救助', '转移', '疏散', '撤离', '安置', '救治', '医疗', '抢救'],
            '物资支援': ['捐赠', '支援', '提供', '分享', '免费', '无偿', '奉献', '物资', '设备', '用品'],
            '志愿服务': ['志愿者', '互助', '团结', '一起', '参与', '服务', '义务', '志愿', '公益'],
            '情感支持': ['加油', '支持', '鼓励', '安慰', '理解', '陪伴', '关心', '温暖', '希望', '信心']
        }
    }
    
    logger.info("聚焦后的行为分类:")
    for category, subcategories in focused_categories.items():
        logger.info(f"\n{category}:")
        for subcategory, keywords in subcategories.items():
            logger.info(f"  {subcategory}: {', '.join(keywords[:5])}...")
    
    return focused_categories

def analyze_data_distribution(df, categories):
    """分析数据分布"""
    logger.info(f"\n=== 数据分布分析 ===")
    
    # 统计各类行为的匹配情况
    distribution_results = {}
    
    for category, subcategories in categories.items():
        category_matches = 0
        subcategory_matches = {}
        
        for subcategory, keywords in subcategories.items():
            subcategory_count = 0
            for keyword in keywords:
                # 统计包含该关键词的微博数量
                matches = df['txt'].str.contains(keyword, na=False, case=False).sum()
                subcategory_count += matches
            
            subcategory_matches[subcategory] = subcategory_count
            category_matches += subcategory_count
        
        distribution_results[category] = {
            'total_matches': category_matches,
            'subcategories': subcategory_matches
        }
    
    # 打印分布结果
    logger.info("行为分类数据分布:")
    for category, results in distribution_results.items():
        logger.info(f"\n{category}:")
        logger.info(f"  总匹配数: {results['total_matches']}")
        logger.info(f"  占比: {results['total_matches']/len(df)*100:.2f}%")
        for subcategory, count in results['subcategories'].items():
            logger.info(f"    {subcategory}: {count} ({count/len(df)*100:.2f}%)")
    
    return distribution_results

def visualize_distribution(distribution_results):
    """可视化数据分布"""
    logger.info(f"\n=== 生成数据分布可视化 ===")
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 主分类分布
    categories = list(distribution_results.keys())
    totals = [results['total_matches'] for results in distribution_results.values()]
    
    ax1.pie(totals, labels=categories, autopct='%1.1f%%', startangle=90)
    ax1.set_title('行为分类总体分布')
    
    # 子分类分布
    subcategories = []
    subcategory_counts = []
    
    for category, results in distribution_results.items():
        for subcategory, count in results['subcategories'].items():
            subcategories.append(f"{category}\n{subcategory}")
            subcategory_counts.append(count)
    
    # 只显示前10个子分类
    top_indices = np.argsort(subcategory_counts)[-10:]
    top_subcategories = [subcategories[i] for i in top_indices]
    top_counts = [subcategory_counts[i] for i in top_indices]
    
    ax2.barh(range(len(top_subcategories)), top_counts)
    ax2.set_yticks(range(len(top_subcategories)))
    ax2.set_yticklabels(top_subcategories)
    ax2.set_xlabel('匹配数量')
    ax2.set_title('子分类分布 (前10)')
    
    plt.tight_layout()
    plt.savefig('data/image/2_behavior_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info("数据分布图已保存到 data/image/2_behavior_distribution_analysis.png")

def create_focused_behavior_script():
    """创建聚焦的行为分析脚本"""
    logger.info("创建聚焦的行为分析脚本...")
    
    script_content = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
聚焦的行为分析脚本
专注于信息搜索行为和亲社会行为的识别
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import logging
from datetime import datetime
import os

# 设置日志
def setup_logger():
    """设置日志记录器"""
    os.makedirs('data/log', exist_ok=True)
    
    logger = logging.getLogger('focused_behavior_analysis')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = f'data/log/focused_behavior_analysis_{timestamp}.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()

class FocusedBehaviorAnalyzer:
    def __init__(self):
        self.behavior_keywords = {{
            '信息搜索行为': {{
                '主动询问': ['请问', '谁知道', '求问', '帮忙', '需要', '了解', '确认', '核实', '求证', '是否', '真假', '真的吗'],
                '关注动态': ['关注', '跟踪', '更新', '最新', '情况', '进展', '建议', '怎么办', '如何', '应该', '推荐', '意见'],
                '信息求证': ['真的吗', '确认', '核实', '求证', '是否', '真假', '靠谱', '可信', '准确'],
                '寻求建议': ['建议', '怎么办', '如何', '应该', '推荐', '意见', '指导', '帮助']
            }},
            '亲社会行为': {{
                '紧急求助': ['救命', '被困', '危险', '紧急', '快', '急', '救命', '求助', '需要帮助', '帮忙', '支持', '援助'],
                '互助支援': ['帮助', '支援', '捐赠', '志愿者', '互助', '团结', '一起', '提供', '分享', '免费', '开放', '可用', '有需要'],
                '情感支持': ['加油', '支持', '鼓励', '安慰', '理解', '陪伴', '感谢', '感动', '温暖', '希望', '信心', '坚强']
            }}
        }}
        self.vectorizer = None
        self.model = None
    
    def create_training_data(self, df, sample_size=2000):
        """创建训练数据（基于聚焦的分类）"""
        logger.info("创建聚焦的训练数据...")
        
        # 随机抽样
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        # 基于聚焦的关键词规则进行标注
        def label_behavior(text):
            text_lower = str(text).lower()
            
            # 计算各类行为的匹配度
            scores = {{
                '信息搜索行为': 0,
                '亲社会行为': 0
            }}
            
            for behavior_type, categories in self.behavior_keywords.items():
                for category, keywords in categories.items():
                    for keyword in keywords:
                        if keyword in text_lower:
                            scores[behavior_type] += 1
            
            # 返回得分最高的行为类型
            if max(scores.values()) == 0:
                return '其他行为'  # 默认分类
            else:
                return max(scores.items(), key=lambda x: x[1])[0]
        
        sample_df['behavior_label'] = sample_df['txt'].apply(label_behavior)
        
        return sample_df
    
    def train_behavior_model(self, training_data):
        """训练行为识别模型"""
        logger.info("训练聚焦的行为识别模型...")
        
        # 准备训练数据
        X = training_data['txt'].fillna('')
        y = training_data['behavior_label']
        
        # 使用TF-IDF特征提取
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        X_tfidf = self.vectorizer.fit_transform(X)
        
        # 训练随机森林模型
        self.model = RandomForestClassifier(
            n_estimators=100,
            criterion='gini',
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = self.model.predict(X_test)
        
        # 计算评估指标
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        evaluation_results = {{
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }}
        
        logger.info(f"聚焦行为识别模型评估结果:")
        logger.info(f"准确率: {precision:.3f}")
        logger.info(f"召回率: {recall:.3f}")
        logger.info(f"F1分数: {f1:.3f}")
        
        return evaluation_results
    
    def predict_behaviors(self, df):
        """预测整个数据集的行为类型"""
        logger.info("预测聚焦的行为类型...")
        
        if self.model is None or self.vectorizer is None:
            raise ValueError("模型未训练，请先调用train_behavior_model方法")
        
        # 特征提取
        X = df['txt'].fillna('')
        X_tfidf = self.vectorizer.transform(X)
        
        # 预测
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # 获取置信度（最大概率值）
        confidence_scores = np.max(probabilities, axis=1)
        
        # 创建结果DataFrame
        results = []
        for idx, (prediction, confidence) in enumerate(zip(predictions, confidence_scores)):
            results.append({{
                'index': idx,
                'predicted_behavior': prediction,
                'behavior_confidence': confidence
            }})
        
        results_df = pd.DataFrame(results)
        
        # 合并原始数据
        df_with_behavior = df.reset_index().merge(results_df, on='index', how='left')
        
        return df_with_behavior

def main():
    """主函数"""
    logger.info("开始聚焦的行为识别分析...")
    
    # 加载数据
    try:
        df = pd.read_csv('data/clean/processed_data.csv')
        logger.info("加载清洗后数据")
    except:
        df = pd.read_excel('data/data_new.xlsx')
        logger.info("加载原始数据")
    
    logger.info(f"数据规模: {len(df)} 条记录")
    
    # 初始化聚焦行为分析器
    analyzer = FocusedBehaviorAnalyzer()
    
    # 创建训练数据
    training_data = analyzer.create_training_data(df)
    
    # 训练模型
    evaluation_results = analyzer.train_behavior_model(training_data)
    
    # 预测整个数据集
    df_with_behavior = analyzer.predict_behaviors(df)
    
    # 保存结果
    os.makedirs('result/behavior', exist_ok=True)
    df_with_behavior.to_csv('result/behavior/focused_behavior_analysis_results.csv', index=False)
    
    # 保存评估结果
    evaluation_df = pd.DataFrame([evaluation_results])
    evaluation_df.to_csv('result/behavior/focused_behavior_evaluation.csv', index=False)
    
    logger.info("聚焦行为识别分析完成！")
    return df_with_behavior, evaluation_results

if __name__ == "__main__":
    main()
'''
    
    with open('focused_behavior_analysis.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    logger.info("聚焦的行为分析脚本已保存到 weibo_3/focused_behavior_analysis.py")

def main():
    """主函数"""
    logger.info("开始分析原始微博数据...")
    
    # 分析原始数据
    df, word_counts = analyze_raw_data()
    
    # 分析行为关键词
    analyze_behavior_keywords()
    
    # 评估当前行为分类
    current_categories = evaluate_behavior_categories()
    
    # 建议聚焦的分类
    focused_categories = suggest_focused_categories()
    
    # 分析数据分布
    distribution_results = analyze_data_distribution(df, focused_categories)
    
    # 可视化分布
    visualize_distribution(distribution_results)
    
    # 创建聚焦的脚本
    create_focused_behavior_script()
    
    logger.info("\n分析完成！")
    logger.info("建议:")
    logger.info("1. 使用聚焦的行为分类：信息搜索行为和亲社会行为")
    logger.info("2. 根据数据分布结果调整关键词")
    logger.info("3. 重点关注匹配度较高的子分类")

if __name__ == "__main__":
    main() 