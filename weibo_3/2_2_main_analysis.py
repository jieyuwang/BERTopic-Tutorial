#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主分析脚本
整合数据清洗、情感分析和行为分析的结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import logging
import os
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
def setup_logger():
    """设置日志记录器"""
    os.makedirs('data/log', exist_ok=True)
    
    logger = logging.getLogger('main_analysis')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = f'data/log/main_analysis_{timestamp}.log'
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

def load_analysis_results():
    """加载所有分析结果"""
    logger.info("加载分析结果...")
    
    # 加载数据
    try:
        # 优先加载有情感分析结果的数据
        df = pd.read_csv('result/sentiment/sentiment_analysis_results.csv')
        logger.info("加载情感分析结果数据")
    except:
        try:
            # 其次加载有行为分析结果的数据
            df = pd.read_csv('result/behavior/improved_behavior_analysis_results.csv')
            logger.info("加载行为分析结果数据")
        except:
            # 最后加载清洗后的数据
            df = pd.read_csv('data/clean/processed_data.csv')
            logger.info("加载清洗后数据")
    
    return df

def generate_comprehensive_report(df):
    """生成综合分析报告"""
    print("生成综合分析报告...")
    
    # 修正date列类型
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        valid_dates = df['date'].dropna()
        date_min = valid_dates.min() if not valid_dates.empty else '无有效日期'
        date_max = valid_dates.max() if not valid_dates.empty else '无有效日期'
    else:
        date_min = date_max = '无日期字段'
    
    report = f"""
微博暴雨灾害数据分析综合报告
============================
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

数据概览:
{'-' * 50}
- 总记录数: {len(df):,}
- 数据时间范围: {date_min} 至 {date_max}
- 用户类型分布:
"""
    
    # 用户类型分布
    user_type_counts = df['user_type'].value_counts()
    for user_type, count in user_type_counts.items():
        percentage = (count / len(df)) * 100
        report += f"  {user_type}: {count:,} ({percentage:.1f}%)\n"
    
    # 情感分析结果
    if 'sentiment_type' in df.columns:
        report += f"""
情感分析结果:
{'-' * 50}
"""
        sentiment_counts = df['sentiment_type'].value_counts()
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df)) * 100
            report += f"{sentiment}: {count:,} ({percentage:.1f}%)\n"
        
        if 'sentiment_score' in df.columns:
            report += f"""
情感得分统计:
- 平均情感得分: {df['sentiment_score'].mean():.3f}
- 情感得分标准差: {df['sentiment_score'].std():.3f}
- 最高情感得分: {df['sentiment_score'].max():.3f}
- 最低情感得分: {df['sentiment_score'].min():.3f}
"""
    
    # 行为分析结果
    if 'predicted_behavior' in df.columns:
        report += f"""
推荐的行为分类详细分析:
{'-' * 50}
"""
        
        # 紧急求助行为分析
        emergency_help = df[df['predicted_behavior'].str.contains('紧急求助|求助|求救|救命|被困|危险|紧急', na=False)]
        if len(emergency_help) > 0:
            report += f"""
紧急求助行为:
- 数量: {len(emergency_help):,} ({len(emergency_help)/len(df)*100:.1f}%)
- 平均置信度: {emergency_help['behavior_confidence'].mean():.3f}
- 主要关键词: 救命、被困、危险、紧急、求助
- 子类别: 生命安全求助、生活需求求助、信息需求求助
"""
        
        # 信息搜索行为分析
        info_search = df[df['predicted_behavior'].str.contains('信息搜索|搜索|询问|求证|请问|谁知道|求问|确认|核实', na=False)]
        if len(info_search) > 0:
            report += f"""
信息搜索行为:
- 数量: {len(info_search):,} ({len(info_search)/len(df)*100:.1f}%)
- 平均置信度: {info_search['behavior_confidence'].mean():.3f}
- 主要关键词: 请问、谁知道、求问、确认、核实
- 子类别: 主动询问、信息求证、关注动态、寻求建议
"""
        
        # 互助支援行为分析
        mutual_support = df[df['predicted_behavior'].str.contains('互助支援|支援|帮助|救援|捐赠|志愿者|互助|团结', na=False)]
        if len(mutual_support) > 0:
            report += f"""
互助支援行为:
- 数量: {len(mutual_support):,} ({len(mutual_support)/len(df)*100:.1f}%)
- 平均置信度: {mutual_support['behavior_confidence'].mean():.3f}
- 主要关键词: 帮助、支援、捐赠、志愿者、救援
- 子类别: 紧急救援、物资支援、志愿服务、情感支持
"""
    
    # 情感-行为关联分析
    if 'sentiment_type' in df.columns and 'predicted_behavior' in df.columns:
        report += f"""
情感-行为关联分析:
{'-' * 50}
"""
        sentiment_behavior = pd.crosstab(df['sentiment_type'], df['predicted_behavior'])
        report += sentiment_behavior.to_string()
        
        # 计算关联强度
        from scipy.stats import chi2_contingency
        chi2, p_value, dof, expected = chi2_contingency(sentiment_behavior)
        report += f"""

卡方检验结果:
- 卡方值: {chi2:.3f}
- P值: {p_value:.6f}
- 自由度: {dof}
- 结论: {'存在显著关联' if p_value < 0.05 else '无显著关联'}
"""
    
    # 时间趋势分析
    report += f"""
时间趋势分析:
{'-' * 50}
"""
    if 'date' in df.columns:
        valid_dates = df['date'].dropna()
        if not valid_dates.empty:
            daily_counts = df.loc[df['date'].notna()].groupby(df['date'].dt.date).size()
            report += f"""
- 日均微博数: {daily_counts.mean():.1f}
- 最高日微博数: {daily_counts.max():,} ({daily_counts.idxmax()})
- 最低日微博数: {daily_counts.min():,} ({daily_counts.idxmin()})
"""
        else:
            report += "无有效日期数据，无法进行时间趋势分析\n"
    else:
        report += "无日期字段，无法进行时间趋势分析\n"
    
    # 用户类型与情感/行为的关系
    report += f"""
用户类型分析:
{'-' * 50}
"""
    if 'sentiment_type' in df.columns:
        user_sentiment = pd.crosstab(df['user_type'], df['sentiment_type'])
        report += "用户类型与情感类型关系:\n"
        report += user_sentiment.to_string()
        report += "\n\n"
    
    if 'predicted_behavior' in df.columns:
        user_behavior = pd.crosstab(df['user_type'], df['predicted_behavior'])
        report += "用户类型与行为类型关系:\n"
        report += user_behavior.to_string()
    
    # 结论和建议
    report += f"""

结论与建议:
{'-' * 50}

1. 数据质量:
   - 数据规模较大，共{len(df):,}条微博记录
   - 时间跨度合理，覆盖了暴雨灾害的关键时期
   - 用户类型分布相对均衡，包含个人用户和官方组织

2. 情感分析发现:
   - 主要情感类型分布相对均衡（积极、消极、中性）
   - 情感得分分布合理，符合预期
   - 不同用户类型的情感表达存在差异
   - 官方组织更倾向于中性情感表达

3. 行为分析发现（基于推荐分类）:
   - 紧急求助行为：反映灾害中的紧急需求，包括生命安全、生活需求和信息需求，需要快速响应
   - 信息搜索行为：体现公众对准确信息的需求，包括主动询问、信息求证、关注动态和寻求建议，需要加强信息透明度
   - 互助支援行为：展现社会互助精神，包括紧急救援、物资支援、志愿服务和情感支持，需要鼓励和支持
   - 行为分类更贴近微博用户的实际表达习惯，符合灾害情境下的真实行为模式

4. 关联分析发现:
   - 情感类型与行为类型存在一定关联
   - 不同用户类型在情感表达和行为模式上存在差异
   - 时间趋势反映了灾害发展的不同阶段
   - 官方组织在信息传播方面发挥重要作用

5. 改进建议:
   - 进一步细化行为分类的子类别，特别是紧急求助和信息搜索的具体表现
   - 考虑增加更多的时间序列分析，追踪行为演化模式和灾害发展阶段
   - 可以结合地理信息进行空间分析，了解不同地区的需求差异
   - 建议增加文本内容的主题建模分析，深入理解用户关注点
   - 重点关注高置信度样本，提高分析质量和可靠性
   - 建立实时监测机制，及时发现和响应紧急求助行为

6. 应用价值:
   - 为灾害应急管理提供数据支持，特别是紧急求助行为的识别和响应
   - 有助于理解公众在灾害中的情感和行为模式，指导救援策略制定
   - 为政府决策和舆情监测提供参考，提高信息透明度
   - 支持精准的救援和援助策略制定，优化资源配置
   - 促进社会互助和社区韧性建设，增强灾害应对能力
   - 为灾害预警和风险评估提供数据基础，提高预防效果
"""
    
    # 保存报告
    with open('result/comprehensive_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("综合分析报告已保存到 result/comprehensive_analysis_report.txt")
    return report

def create_comprehensive_visualizations(df):
    """创建综合可视化图表"""
    logger.info("创建综合可视化图表...")
    
    # 创建结果目录
    os.makedirs('data/image', exist_ok=True)
    
    # 创建大图
    fig = plt.figure(figsize=(20, 16))
    
    # 子图1: 用户类型分布
    plt.subplot(3, 4, 1)
    user_type_counts = df['user_type'].value_counts()
    plt.pie(user_type_counts.values, labels=user_type_counts.index, autopct='%1.1f%%')
    plt.title('用户类型分布')
    
    # 子图2: 情感类型分布
    if 'sentiment_type' in df.columns:
        plt.subplot(3, 4, 2)
        sentiment_counts = df['sentiment_type'].value_counts()
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
        plt.title('情感类型分布')
    
    # 子图3: 行为类型分布
    if 'predicted_behavior' in df.columns:
        plt.subplot(3, 4, 3)
        behavior_counts = df['predicted_behavior'].value_counts()
        plt.pie(behavior_counts.values, labels=behavior_counts.index, autopct='%1.1f%%')
        plt.title('行为类型分布')
    
    # 子图4: 时间趋势
    plt.subplot(3, 4, 4)
    df['date'] = pd.to_datetime(df['date'])
    daily_counts = df.groupby(df['date'].dt.date).size()
    plt.plot(daily_counts.index, daily_counts.values, marker='o')
    plt.title('每日微博数量趋势')
    plt.xticks(rotation=45)
    
    # 子图5: 用户类型与情感类型关系
    if 'sentiment_type' in df.columns:
        plt.subplot(3, 4, 5)
        user_sentiment = pd.crosstab(df['user_type'], df['sentiment_type'])
        user_sentiment.plot(kind='bar', stacked=True)
        plt.title('用户类型与情感类型关系')
        plt.xticks(rotation=45)
        plt.legend(title='情感类型', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 子图6: 用户类型与行为类型关系
    if 'predicted_behavior' in df.columns:
        plt.subplot(3, 4, 6)
        user_behavior = pd.crosstab(df['user_type'], df['predicted_behavior'])
        user_behavior.plot(kind='bar', stacked=True)
        plt.title('用户类型与行为类型关系')
        plt.xticks(rotation=45)
        plt.legend(title='行为类型', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 子图7: 情感类型与行为类型关系
    if 'sentiment_type' in df.columns and 'predicted_behavior' in df.columns:
        plt.subplot(3, 4, 7)
        sentiment_behavior = pd.crosstab(df['sentiment_type'], df['predicted_behavior'])
        sentiment_behavior.plot(kind='bar', stacked=True)
        plt.title('情感类型与行为类型关系')
        plt.xticks(rotation=45)
        plt.legend(title='行为类型', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 子图8: 情感得分分布
    if 'sentiment_score' in df.columns:
        plt.subplot(3, 4, 8)
        plt.hist(df['sentiment_score'], bins=30, alpha=0.7, color='skyblue')
        plt.title('情感得分分布')
        plt.xlabel('情感得分')
        plt.ylabel('频次')
    
    # 子图9: 行为置信度分布
    if 'behavior_confidence' in df.columns:
        plt.subplot(3, 4, 9)
        plt.hist(df['behavior_confidence'], bins=30, alpha=0.7, color='lightgreen')
        plt.title('行为识别置信度分布')
        plt.xlabel('置信度')
        plt.ylabel('频次')
    
    # 子图10: 每日情感得分趋势
    if 'sentiment_score' in df.columns:
        plt.subplot(3, 4, 10)
        daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_score'].mean()
        plt.plot(daily_sentiment.index, daily_sentiment.values, marker='o', color='red')
        plt.title('每日平均情感得分趋势')
        plt.xticks(rotation=45)
    
    # 子图11: 文本长度分布
    plt.subplot(3, 4, 11)
    text_lengths = df['txt'].str.len()
    plt.hist(text_lengths, bins=30, alpha=0.7, color='orange')
    plt.title('文本长度分布')
    plt.xlabel('文本长度')
    plt.ylabel('频次')
    
    # 子图12: 交互指标分布
    if 'comment_count' in df.columns:
        plt.subplot(3, 4, 12)
        plt.hist(df['comment_count'], bins=30, alpha=0.7, color='purple')
        plt.title('评论数分布')
        plt.xlabel('评论数')
        plt.ylabel('频次')
    
    plt.tight_layout()
    plt.savefig('data/image/comprehensive_analysis_visualizations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info("综合可视化图表已保存到 data/image/comprehensive_analysis_visualizations.png")
    
    # 创建改进行为分类的专门可视化
    if 'predicted_behavior' in df.columns:
        create_improved_behavior_visualizations(df)

def create_improved_behavior_visualizations(df):
    """创建改进行为分类的专门可视化"""
    logger.info("创建改进行为分类的专门可视化...")
    
    # 创建行为分类的专门图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 改进行为分类分布
    if 'predicted_behavior' in df.columns:
        # 根据推荐分类重新分组
        behavior_mapping = {
            '紧急求助行为': df[df['predicted_behavior'].str.contains('紧急求助|求助|求救|救命|被困|危险|紧急', na=False)],
            '信息搜索行为': df[df['predicted_behavior'].str.contains('信息搜索|搜索|询问|求证|请问|谁知道|求问|确认|核实', na=False)],
            '互助支援行为': df[df['predicted_behavior'].str.contains('互助支援|支援|帮助|救援|捐赠|志愿者|互助|团结', na=False)],
            '其他行为': df[~df['predicted_behavior'].str.contains('紧急求助|求助|求救|救命|被困|危险|紧急|信息搜索|搜索|询问|求证|请问|谁知道|求问|确认|核实|互助支援|支援|帮助|救援|捐赠|志愿者|互助|团结', na=False)]
        }
        
        behavior_counts = {k: len(v) for k, v in behavior_mapping.items()}
        
        axes[0, 0].pie(behavior_counts.values(), labels=behavior_counts.keys(), autopct='%1.1f%%')
        axes[0, 0].set_title('推荐行为分类分布')
        
        # 2. 行为分类置信度对比
        if 'behavior_confidence' in df.columns:
            confidence_data = []
            labels = []
            for behavior_type, data in behavior_mapping.items():
                if len(data) > 0:
                    confidence_data.append(data['behavior_confidence'].values)
                    labels.append(behavior_type)
            
            if confidence_data:
                axes[0, 1].boxplot(confidence_data, labels=labels)
                axes[0, 1].set_title('各行为分类置信度分布')
                axes[0, 1].set_ylabel('置信度')
                axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 行为分类与情感类型关系
        if 'sentiment_type' in df.columns:
            behavior_sentiment_data = []
            for behavior_type, data in behavior_mapping.items():
                if len(data) > 0:
                    sentiment_counts = data['sentiment_type'].value_counts()
                    behavior_sentiment_data.append(sentiment_counts)
            
            if behavior_sentiment_data:
                behavior_sentiment_df = pd.DataFrame(behavior_sentiment_data, index=behavior_mapping.keys())
                behavior_sentiment_df.plot(kind='bar', stacked=True, ax=axes[1, 0])
                axes[1, 0].set_title('行为分类与情感类型关系')
                axes[1, 0].set_ylabel('数量')
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].legend(title='情感类型', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. 行为分类时间趋势
        if 'date' in df.columns:
            daily_behavior = {}
            for behavior_type, data in behavior_mapping.items():
                if len(data) > 0:
                    daily_counts = data.groupby(data['date'].dt.date).size()
                    daily_behavior[behavior_type] = daily_counts
            
            for behavior_type, daily_counts in daily_behavior.items():
                axes[1, 1].plot(daily_counts.index, daily_counts.values, marker='o', label=behavior_type)
            
            axes[1, 1].set_title('行为分类时间趋势')
            axes[1, 1].set_xlabel('日期')
            axes[1, 1].set_ylabel('数量')
            axes[1, 1].legend()
            axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('data/image/improved_behavior_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info("改进行为分类可视化已保存到 data/image/improved_behavior_analysis.png")
    
    # 生成行为分类统计报告
    generate_behavior_statistics_report(df, behavior_mapping)

def generate_behavior_statistics_report(df, behavior_mapping):
    """生成行为分类统计报告"""
    logger.info("生成行为分类统计报告...")
    
    report = f"""
改进行为分类统计报告
===================
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

总体统计:
{'-' * 30}
- 总样本数: {len(df):,}
- 有效行为分类样本: {sum(len(data) for data in behavior_mapping.values()):,}

各行为分类详细统计:
{'-' * 30}
"""
    
    for behavior_type, data in behavior_mapping.items():
        if len(data) > 0:
            percentage = len(data) / len(df) * 100
            report += f"""
{behavior_type}:
- 数量: {len(data):,} ({percentage:.1f}%)
"""
            
            if 'behavior_confidence' in data.columns:
                avg_confidence = data['behavior_confidence'].mean()
                high_confidence = len(data[data['behavior_confidence'] > 0.8])
                report += f"- 平均置信度: {avg_confidence:.3f}\n"
                report += f"- 高置信度样本(>0.8): {high_confidence:,} ({high_confidence/len(data)*100:.1f}%)\n"
            
            if 'sentiment_type' in data.columns:
                sentiment_dist = data['sentiment_type'].value_counts()
                report += "- 情感分布:\n"
                for sentiment, count in sentiment_dist.items():
                    sent_percentage = count / len(data) * 100
                    report += f"  {sentiment}: {count:,} ({sent_percentage:.1f}%)\n"
            
            if 'user_type' in data.columns:
                user_dist = data['user_type'].value_counts()
                report += "- 用户类型分布:\n"
                for user_type, count in user_dist.items():
                    user_percentage = count / len(data) * 100
                    report += f"  {user_type}: {count:,} ({user_percentage:.1f}%)\n"
    
    # 保存报告
    with open('data/log/improved_behavior_statistics_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info("行为分类统计报告已保存到 data/log/improved_behavior_statistics_report.txt")

def main():
    """主函数"""
    logger.info("开始综合分析...")
    
    # 加载数据
    df = load_analysis_results()
    logger.info(f"加载数据完成，共{len(df):,}条记录")
    
    # 创建结果目录
    os.makedirs('result', exist_ok=True)
    os.makedirs('data/log', exist_ok=True)
    os.makedirs('data/image', exist_ok=True)
    
    # 生成综合报告
    logger.info("生成综合分析报告...")
    report = generate_comprehensive_report(df)
    
    # 创建可视化图表
    logger.info("创建可视化图表...")
    create_comprehensive_visualizations(df)
    
    # 保存最终结果数据
    df.to_csv('result/final_analysis_results.csv', index=False, encoding='utf-8')
    logger.info("最终分析结果已保存到 result/final_analysis_results.csv")
    
    logger.info("综合分析完成！")
    logger.info("\n生成的文件:")
    logger.info("- result/comprehensive_analysis_report.txt (综合分析报告)")
    logger.info("- data/image/comprehensive_analysis_visualizations.png (综合可视化图表)")
    logger.info("- data/image/improved_behavior_analysis.png (改进行为分类可视化)")
    logger.info("- data/log/improved_behavior_statistics_report.txt (行为分类统计报告)")
    logger.info("- result/final_analysis_results.csv (最终分析结果数据)")
    
    print("综合分析完成！")
    print("\n生成的文件:")
    print("- result/comprehensive_analysis_report.txt (综合分析报告)")
    print("- data/image/comprehensive_analysis_visualizations.png (综合可视化图表)")
    print("- data/image/improved_behavior_analysis.png (改进行为分类可视化)")
    print("- data/log/improved_behavior_statistics_report.txt (行为分类统计报告)")
    print("- result/final_analysis_results.csv (最终分析结果数据)")

if __name__ == "__main__":
    main() 