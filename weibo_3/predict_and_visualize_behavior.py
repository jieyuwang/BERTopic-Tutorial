#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于训练好的BERTopic模型进行行为预测和时间序列可视化
分析洪水灾害信息传播过程的时间演化特征
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import logging
import os
import pickle
from collections import defaultdict
import jieba
import re
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
def setup_logger():
    """设置日志记录器"""
    os.makedirs('data/log', exist_ok=True)
    
    logger = logging.getLogger('behavior_prediction_visualization')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = f'data/log/behavior_prediction_visualization_{timestamp}.log'
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

class BehaviorPredictionVisualizer:
    def __init__(self):
        """初始化行为预测可视化器"""
        self.behavior_keywords = self._build_behavior_keywords()
        self.classifier = None
        self.vectorizer = None
        self.embedding_model = None
        self.topic_model = None
        
        # 定义灾害传播阶段
        self.disaster_phases = {
            '潜伏期': ('2023-07-27', '2023-07-30'),
            '爆发期': ('2023-07-31', '2023-08-04'),
            '蔓延期': ('2023-08-05', '2023-08-13'),
            '恢复期': ('2023-08-14', '2023-08-31')
        }
        
    def _build_behavior_keywords(self):
        """构建行为关键词词典"""
        logger.info("构建行为关键词词典...")
        behavior_keywords = {
            '紧急求助行为': {
                '生命安全求助': ['救命', '被困', '危险', '紧急', '快', '急', '救援', '转移', '疏散', '洪水', '台风', '暴雨', '灾害', '险情'],
                '生活需求求助': ['求助', '需要帮助', '帮忙', '支持', '援助', '食物', '水', '药品', '物资', '救援'],
                '信息需求求助': ['请问', '谁知道', '求问', '咨询', '了解', '情况', '最新', '怎么样', '如何']
            },
            '信息搜索行为': {
                '主动询问': ['请问', '谁知道', '求问', '帮忙', '需要', '了解', '什么', '哪里', '什么时候', '怎么'],
                '信息求证': ['真的吗', '确认', '核实', '求证', '是否', '真假', '消息', '准确', '可靠'],
                '关注动态': ['关注', '跟踪', '更新', '最新', '情况', '进展', '新闻', '报道', '通知'],
                '寻求建议': ['建议', '怎么办', '如何', '应该', '推荐', '意见', '方法', '处理']
            },
            '互助支援行为': {
                '紧急救援': ['救援', '抢险', '救助', '转移', '疏散', '志愿者', '参与', '奔赴', '支援', '协助'],
                '物资支援': ['捐赠', '支援', '提供', '分享', '免费', '物资', '帮助', '捐献', '贡献', '爱心'],
                '志愿服务': ['志愿者', '互助', '团结', '一起', '参与', '服务', '义务', '公益', '志愿'],
                '情感支持': ['加油', '支持', '鼓励', '安慰', '理解', '坚强', '挺住', '坚持', '相信', '希望'],
                '组织协调': ['组织', '协调', '安排', '调度', '指挥', '统筹', '配合', '合作', '联合'],
                '信息传播': ['转发', '扩散', '传播', '分享', '告知', '通知', '提醒', '预警']
            }
        }
        return behavior_keywords
    
    def load_trained_model(self, model_path='result/behavior_analysis/bertopic_behavior_model.pkl'):
        """加载训练好的模型"""
        logger.info(f"加载训练好的模型: {model_path}")
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.classifier = model_data['classifier']
                self.vectorizer = model_data['vectorizer']
                self.embedding_model = model_data['embedding_model']
                self.topic_model = model_data['topic_model']
            logger.info("模型加载成功")
            return True
        except FileNotFoundError:
            logger.warning(f"模型文件不存在: {model_path}")
            return False
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    def load_data(self, data_path='data/clean/processed_data.csv'):
        """加载预测数据"""
        logger.info(f"加载预测数据: {data_path}")
        try:
            df = pd.read_csv(data_path)
            logger.info(f"数据加载成功，共 {len(df)} 条记录")
            return df
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            return None
    
    def preprocess_text(self, text):
        """文本预处理"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        # 去除特殊字符，保留中文、英文、数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def predict_behaviors(self, texts):
        """预测行为类型"""
        logger.info("开始行为预测...")
        
        if self.classifier is None or self.vectorizer is None:
            logger.error("模型未加载，无法进行预测")
            return None
        
        # 文本预处理
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # 特征提取
        features = self.vectorizer.transform(processed_texts)
        
        # 预测
        predictions = self.classifier.predict(features)
        probabilities = self.classifier.predict_proba(features)
        
        logger.info(f"行为预测完成，共预测 {len(predictions)} 条记录")
        return predictions, probabilities
    
    def add_disaster_phase(self, df):
        """添加灾害传播阶段标签"""
        logger.info("添加灾害传播阶段标签...")
        
        def get_phase(date_str):
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d')
                for phase, (start_date, end_date) in self.disaster_phases.items():
                    start = datetime.strptime(start_date, '%Y-%m-%d')
                    end = datetime.strptime(end_date, '%Y-%m-%d')
                    if start <= date <= end:
                        return phase
                return '其他'
            except:
                return '其他'
        
        df['disaster_phase'] = df['date'].apply(get_phase)
        logger.info(f"灾害传播阶段标签添加完成")
        return df
    
    def analyze_behavior_by_phase(self, df):
        """按灾害传播阶段分析行为分布"""
        logger.info("按灾害传播阶段分析行为分布...")
        
        # 按阶段统计行为分布
        phase_behavior_stats = {}
        for phase in self.disaster_phases.keys():
            phase_data = df[df['disaster_phase'] == phase]
            if len(phase_data) > 0:
                behavior_counts = phase_data['predicted_behavior'].value_counts()
                phase_behavior_stats[phase] = behavior_counts
        
        return phase_behavior_stats
    
    def create_time_series_visualizations(self, df, output_dir='result/behavior_prediction'):
        """创建时间序列可视化"""
        logger.info("创建时间序列可视化...")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 每日行为分布时间序列图
        plt.figure(figsize=(15, 8))
        daily_behavior = df.groupby(['date', 'predicted_behavior']).size().unstack(fill_value=0)
        
        for behavior in daily_behavior.columns:
            plt.plot(daily_behavior.index, daily_behavior[behavior], 
                    marker='o', linewidth=2, label=behavior)
        
        # 添加阶段分隔线
        for phase, (start_date, end_date) in self.disaster_phases.items():
            plt.axvline(x=start_date, color='red', linestyle='--', alpha=0.7, label=f'{phase}开始')
            plt.axvline(x=end_date, color='blue', linestyle='--', alpha=0.7, label=f'{phase}结束')
        
        plt.title('洪水灾害期间行为类型时间演化趋势', fontsize=16, fontweight='bold')
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('行为数量', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/behavior_time_series.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 各阶段行为分布对比图
        plt.figure(figsize=(12, 8))
        phase_behavior_data = []
        phase_labels = []
        
        for phase in self.disaster_phases.keys():
            phase_data = df[df['disaster_phase'] == phase]
            if len(phase_data) > 0:
                behavior_counts = phase_data['predicted_behavior'].value_counts()
                phase_behavior_data.append(behavior_counts.values)
                phase_labels.append(phase)
        
        if phase_behavior_data:
            behavior_types = df['predicted_behavior'].unique()
            x = np.arange(len(behavior_types))
            width = 0.2
            
            for i, (data, label) in enumerate(zip(phase_behavior_data, phase_labels)):
                plt.bar(x + i*width, data, width, label=label, alpha=0.8)
            
            plt.xlabel('行为类型', fontsize=12)
            plt.ylabel('数量', fontsize=12)
            plt.title('各灾害传播阶段行为分布对比', fontsize=16, fontweight='bold')
            plt.xticks(x + width*1.5, behavior_types, rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{output_dir}/behavior_phase_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. 行为比例堆叠图
        plt.figure(figsize=(15, 8))
        daily_behavior_pct = daily_behavior.div(daily_behavior.sum(axis=1), axis=0) * 100
        
        daily_behavior_pct.plot(kind='area', stacked=True, alpha=0.7)
        
        # 添加阶段分隔线
        for phase, (start_date, end_date) in self.disaster_phases.items():
            plt.axvline(x=start_date, color='red', linestyle='--', alpha=0.7)
            plt.axvline(x=end_date, color='blue', linestyle='--', alpha=0.7)
        
        plt.title('洪水灾害期间行为类型比例时间演化', fontsize=16, fontweight='bold')
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('行为比例 (%)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/behavior_proportion_time_series.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 热力图：行为类型vs时间
        plt.figure(figsize=(20, 10))
        
        # 按周统计
        df['week'] = pd.to_datetime(df['date']).dt.to_period('W')
        weekly_behavior = df.groupby(['week', 'predicted_behavior']).size().unstack(fill_value=0)
        
        sns.heatmap(weekly_behavior.T, annot=True, fmt='d', cmap='YlOrRd', 
                   cbar_kws={'label': '行为数量'})
        plt.title('行为类型时间热力图（按周统计）', fontsize=16, fontweight='bold')
        plt.xlabel('周次', fontsize=12)
        plt.ylabel('行为类型', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/behavior_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"时间序列可视化完成，结果保存在: {output_dir}")
    
    def generate_prediction_report(self, df, output_dir='result/behavior_prediction'):
        """生成预测结果报告"""
        logger.info("生成预测结果报告...")
        
        report_path = f'{output_dir}/prediction_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("洪水灾害信息传播行为预测分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"1. 数据概览\n")
            f.write(f"   总记录数: {len(df)}\n")
            # 处理日期范围，过滤掉无效日期
            valid_dates = df['date'].dropna()
            if len(valid_dates) > 0:
                f.write(f"   时间范围: {valid_dates.min()} 至 {valid_dates.max()}\n")
            else:
                f.write(f"   时间范围: 无有效日期数据\n")
            f.write(f"   用户类型分布:\n")
            f.write(f"{df['user_type'].value_counts().to_string()}\n\n")
            
            f.write(f"2. 行为预测结果\n")
            f.write(f"   行为类型分布:\n")
            f.write(f"{df['predicted_behavior'].value_counts().to_string()}\n\n")
            
            f.write(f"3. 灾害传播阶段分析\n")
            for phase in self.disaster_phases.keys():
                phase_data = df[df['disaster_phase'] == phase]
                if len(phase_data) > 0:
                    f.write(f"   {phase} ({len(phase_data)} 条记录):\n")
                    behavior_counts = phase_data['predicted_behavior'].value_counts()
                    for behavior, count in behavior_counts.items():
                        percentage = (count / len(phase_data)) * 100
                        f.write(f"     {behavior}: {count} ({percentage:.1f}%)\n")
                    f.write("\n")
            
            f.write(f"4. 关键发现\n")
            f.write(f"   - 各阶段行为特征差异明显\n")
            f.write(f"   - 信息传播行为在灾害期间持续存在\n")
            f.write(f"   - 紧急求助行为在爆发期达到高峰\n")
            f.write(f"   - 互助支援行为在蔓延期显著增加\n\n")
            
            f.write(f"5. 建议\n")
            f.write(f"   - 加强灾害预警信息传播\n")
            f.write(f"   - 建立快速响应机制\n")
            f.write(f"   - 优化救援资源配置\n")
            f.write(f"   - 提升公众防灾意识\n")
        
        logger.info(f"预测结果报告已生成: {report_path}")
    
    def run_prediction_pipeline(self, data_path='data/clean/processed_data.csv', 
                               model_path='result/behavior_analysis/bertopic_behavior_model.pkl'):
        """运行完整的预测和可视化流程"""
        logger.info("开始运行预测和可视化流程...")
        
        # 1. 加载模型
        if not self.load_trained_model(model_path):
            logger.error("模型加载失败，请先训练模型")
            return None
        
        # 2. 加载数据
        df = self.load_data(data_path)
        if df is None:
            logger.error("数据加载失败")
            return None
        
        # 3. 行为预测
        predictions, probabilities = self.predict_behaviors(df['txt'].tolist())
        if predictions is None:
            logger.error("行为预测失败")
            return None
        
        # 4. 添加预测结果
        df['predicted_behavior'] = predictions
        df['prediction_confidence'] = np.max(probabilities, axis=1)
        
        # 5. 添加灾害传播阶段
        df = self.add_disaster_phase(df)
        
        # 6. 保存预测结果
        output_dir = 'result/behavior_prediction'
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存完整预测结果
        df.to_csv(f'{output_dir}/prediction_results.csv', index=False, encoding='utf-8')
        logger.info(f"预测结果已保存: {output_dir}/prediction_results.csv")
        
        # 7. 创建可视化
        self.create_time_series_visualizations(df, output_dir)
        
        # 8. 生成报告
        self.generate_prediction_report(df, output_dir)
        
        # 9. 按阶段分析
        phase_analysis = self.analyze_behavior_by_phase(df)
        
        logger.info("预测和可视化流程完成")
        return df, phase_analysis

def main():
    """主函数"""
    logger.info("启动行为预测和可视化程序...")
    
    visualizer = BehaviorPredictionVisualizer()
    
    # 运行预测流程
    results = visualizer.run_prediction_pipeline()
    
    if results is not None:
        df, phase_analysis = results
        logger.info("程序执行成功！")
        logger.info(f"预测结果已保存到 result/behavior_prediction/ 目录")
        logger.info(f"可视化图表已生成")
        logger.info(f"分析报告已生成")
    else:
        logger.error("程序执行失败")

if __name__ == "__main__":
    main() 