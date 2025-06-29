#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
微博数据BERTopic主题建模完整流程运行脚本
按顺序执行：数据预处理 -> 词汇分析 -> 分词 -> 嵌入向量生成 -> 主题建模 -> 评估分析
"""

import os
import sys
import subprocess
import time
import shutil

def run_step(step_name, script_path, description, timeout=None):
    """运行单个处理步骤"""
    print(f"\n{'='*60}")
    print(f"步骤: {step_name}")
    print(f"描述: {description}")
    print(f"脚本: {script_path}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # 运行脚本
        cmd = [sys.executable, script_path]
        result = subprocess.run(cmd, 
                              capture_output=True, 
                              text=True, 
                              encoding='utf-8',
                              timeout=timeout)
        
        # 打印输出
        if result.stdout:
            print("输出:")
            print(result.stdout)
        
        if result.stderr:
            print("错误:")
            print(result.stderr)
        
        # 检查是否成功
        if result.returncode == 0:
            elapsed_time = time.time() - start_time
            print(f"✅ {step_name} 完成 (耗时: {elapsed_time:.2f}秒)")
            return True
        else:
            print(f"❌ {step_name} 失败 (返回码: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ {step_name} 超时")
        return False
    except Exception as e:
        print(f"❌ {step_name} 执行异常: {e}")
        return False

def check_file_exists(file_path, description):
    """检查文件是否存在"""
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        size_str = f"({file_size:,} bytes)" if file_size < 1024*1024 else f"({file_size/1024/1024:.1f} MB)"
        print(f"✅ {description}: {file_path} {size_str}")
        return True
    else:
        print(f"❌ {description}: {file_path} (文件不存在)")
        return False

def create_directories():
    """创建必要的目录"""
    directories = [
        "data",
        "results", 
        "vocabulary_analysis",
        "embedding",
        "分词"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 确保目录存在: {directory}")

def backup_userdict():
    """备份用户自定义词典"""
    original_file = "分词/userdict.txt"
    backup_file = "分词/userdict_backup.txt"
    
    if os.path.exists(original_file):
        shutil.copy2(original_file, backup_file)
        print(f"📋 备份用户词典: {backup_file}")

def main():
    """主函数"""
    print("=== 微博数据BERTopic主题建模完整流程 ===")
    print("开始时间:", time.strftime("%Y-%m-%d %H:%M:%S"))
    
    # 创建必要目录
    print("\n创建必要目录...")
    create_directories()
    
    # 备份用户词典
    backup_userdict()
    
    # 检查输入文件
    print("\n检查输入文件...")
    input_file = "data/data_new.xlsx"
    if not check_file_exists(input_file, "输入Excel文件"):
        print("请确保data_new.xlsx文件存在于data目录中")
        return
    
    # 步骤1: 数据预处理
    step1_success = run_step(
        "数据预处理",
        "1_data_preprocessing.py",
        "清理Excel数据，提取文本和时间信息"
    )
    
    if not step1_success:
        print("数据预处理失败，停止执行")
        return
    
    # 检查预处理结果
    print("\n检查预处理结果...")
    check_file_exists("data/文本.txt", "清理后的文本文件")
    check_file_exists("data/时间.txt", "时间信息文件")
    check_file_exists("data/weibo_clean_data.csv", "完整清理数据")
    
    # 步骤2: 词汇分析（新增）
    step2_success = run_step(
        "词汇分析",
        "2_analyze_vocabulary.py",
        "分析高频词汇，生成用户自定义词典建议"
    )
    
    if not step2_success:
        print("词汇分析失败，但继续执行后续步骤")
    else:
        # 检查词汇分析结果
        print("\n检查词汇分析结果...")
        check_file_exists("vocabulary_analysis/wordcloud.png", "词汇云图")
        check_file_exists("分词/userdict_updated.txt", "更新后的用户词典")
    
    # 步骤3: 分词处理
    step3_success = run_step(
        "分词处理",
        "3_word_segmentation.py",
        "对文本进行中文分词"
    )
    
    if not step3_success:
        print("分词处理失败，停止执行")
        return
    
    # 检查分词结果
    print("\n检查分词结果...")
    check_file_exists("data/切词.txt", "分词结果文件")
    check_file_exists("data/切词_word_freq.csv", "词频统计文件")
    
    # 步骤4: 嵌入向量生成
    print("\n注意: 嵌入向量生成可能需要较长时间，请耐心等待...")
    step4_success = run_step(
        "嵌入向量生成",
        "4_generate_embeddings.py",
        "生成文本嵌入向量",
        timeout=3600  # 1小时超时
    )
    
    if not step4_success:
        print("嵌入向量生成失败，停止执行")
        return
    
    # 检查嵌入向量结果
    print("\n检查嵌入向量结果...")
    check_file_exists("data/embedding_sen.npy", "Sentence-Transformers嵌入向量")
    check_file_exists("data/embedding_bert.npy", "BERT嵌入向量")
    check_file_exists("data/embedding_info.csv", "嵌入向量信息")
    
    # 步骤5: 主题建模
    step5_success = run_step(
        "主题建模",
        "5_main_weibo.py",
        "使用BERTopic进行主题建模",
        timeout=1800  # 30分钟超时
    )
    
    if not step5_success:
        print("主题建模失败")
        return
    
    # 步骤6: 评估分析（新增）
    step6_success = run_step(
        "评估分析",
        "6_evaluation_metrics.py",
        "计算主题建模评估指标"
    )
    
    if not step6_success:
        print("评估分析失败，但流程已完成")
    
    # 检查最终结果
    print("\n检查最终结果...")
    check_file_exists("results/weibo_clustering_results.csv", "基础聚类结果")
    check_file_exists("results/weibo_complete_results.csv", "完整聚类结果")
    check_file_exists("results/weibo_topic_info.csv", "主题信息")
    check_file_exists("results/weibo_analysis_stats.csv", "分析统计")
    check_file_exists("results/weibo_evaluation_metrics.csv", "评估指标")
    
    # 完成
    print(f"\n{'='*60}")
    print("🎉 所有步骤执行完成！")
    print("结束时间:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"{'='*60}")
    
    print("\n📁 结果文件位置:")
    print("- 数据文件: data/")
    print("- 聚类结果: results/")
    print("- 词汇分析: vocabulary_analysis/")
    print("- 嵌入向量: embedding/")
    
    print("\n📊 主要输出文件:")
    print("- results/weibo_complete_results.csv: 完整的聚类结果（包含原文和时间）")
    print("- results/weibo_topic_info.csv: 主题详细信息")
    print("- results/weibo_analysis_stats.csv: 分析统计信息")
    print("- results/weibo_evaluation_metrics.csv: 评估指标")
    print("- vocabulary_analysis/wordcloud.png: 词汇云图")
    print("- 分词/userdict_updated.txt: 更新后的用户词典")
    
    print("\n💡 建议:")
    print("- 查看词汇分析结果，根据需要调整用户词典")
    print("- 检查主题质量，可能需要调整BERTopic参数")
    print("- 分析评估指标，优化模型性能")

if __name__ == "__main__":
    main() 