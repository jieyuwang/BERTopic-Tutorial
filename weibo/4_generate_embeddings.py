#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
嵌入向量生成脚本 - 使用Sentence Transformers生成文本嵌入向量
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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

def load_model(model_name='paraphrase-multilingual-MiniLM-L12-v2'):
    """加载预训练模型"""
    print(f"正在加载模型: {model_name}")
    print("注意：首次加载可能需要下载模型文件，请耐心等待...")
    
    # 模型选择建议
    model_suggestions = {
        'fast': 'all-MiniLM-L6-v2',  # 最快，但质量稍低
        'balanced': 'paraphrase-multilingual-MiniLM-L12-v2',  # 平衡
        'quality': 'paraphrase-multilingual-mpnet-base-v2'  # 最高质量，但较慢
    }
    
    print("模型选择建议:")
    for speed, model in model_suggestions.items():
        print(f"  - {speed}: {model}")
    
    try:
        print("开始下载/加载模型...")
        model = SentenceTransformer(model_name)
        print(f"✓ 模型加载成功！")
        print(f"✓ 模型维度: {model.get_sentence_embedding_dimension()}")
        return model
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        print("可能的解决方案：")
        print("1. 检查网络连接")
        print("2. 尝试使用其他模型，如 'all-MiniLM-L6-v2'")
        print("3. 手动下载模型到本地")
        
        # 尝试使用备用模型
        backup_model = 'all-MiniLM-L6-v2'
        print(f"尝试使用备用模型: {backup_model}")
        try:
            model = SentenceTransformer(backup_model)
            print(f"✓ 备用模型加载成功！")
            print(f"✓ 模型维度: {model.get_sentence_embedding_dimension()}")
            return model
        except Exception as e2:
            print(f"✗ 备用模型也失败: {e2}")
            return None

def load_data():
    """加载分词后的数据"""
    print("正在加载数据...")
    
    try:
        # 尝试加载分词后的CSV文件
        print("尝试加载CSV文件: data/weibo_segmented_data.csv")
        df = pd.read_csv('data/weibo_segmented_data.csv')
        print(f"✓ 成功加载分词数据，共 {len(df):,} 行")
        return df
    except Exception as e1:
        print(f"CSV文件加载失败: {e1}")
        try:
            # 如果CSV不存在，尝试加载txt文件
            print("尝试加载TXT文件: data/切词.txt")
            with open('data/切词.txt', 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            print(f"✓ 成功加载分词文本，共 {len(texts):,} 行")
            return texts
        except Exception as e2:
            print(f"✗ 所有数据加载方式都失败:")
            print(f"  - CSV错误: {e1}")
            print(f"  - TXT错误: {e2}")
            return None

def generate_embeddings(model, texts, batch_size=64):
    """生成文本嵌入向量"""
    print(f"\n=== 开始生成嵌入向量 ===")
    print(f"文本数量: {len(texts):,}")
    print(f"批次大小: {batch_size}")
    print(f"预计批次数: {len(texts) // batch_size + 1}")
    
    # 过滤空文本
    print("正在过滤空文本...")
    valid_texts = [text for text in texts if text and text.strip()]
    print(f"✓ 有效文本数量: {len(valid_texts):,}")
    
    if not valid_texts:
        print("✗ 没有有效的文本数据！")
        return None
    
    # 生成嵌入向量
    print("开始生成嵌入向量...")
    embeddings = []
    total_batches = (len(valid_texts) + batch_size - 1) // batch_size
    
    # 减少进度打印频率，每50个批次打印一次
    print_frequency = 50
    
    # 添加时间监控
    import time
    start_time = time.time()
    
    for i in range(0, len(valid_texts), batch_size):
        batch_num = i // batch_size + 1
        batch_texts = valid_texts[i:i+batch_size]
        
        # 减少进度打印频率
        if batch_num % print_frequency == 0 or batch_num <= 5 or batch_num >= total_batches - 5:
            elapsed_time = time.time() - start_time
            print(f"处理批次 {batch_num}/{total_batches} ({i+len(batch_texts):,}/{len(valid_texts):,}) - 已用时: {elapsed_time:.1f}秒")
        
        batch_start_time = time.time()
        try:
            batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
            embeddings.append(batch_embeddings)
            
            batch_time = time.time() - batch_start_time
            
            # 只在特定批次显示完成信息
            if batch_num % print_frequency == 0 or batch_num <= 5 or batch_num >= total_batches - 5:
                print(f"  ✓ 批次 {batch_num} 完成，形状: {batch_embeddings.shape}，用时: {batch_time:.2f}秒")
                
        except Exception as e:
            print(f"  ✗ 批次 {batch_num} 失败: {e}")
            # 如果批次失败，尝试逐个处理
            print("  尝试逐个处理...")
            batch_embeddings = []
            for j, text in enumerate(batch_texts):
                try:
                    single_embedding = model.encode([text], show_progress_bar=False)
                    batch_embeddings.append(single_embedding[0])
                    if (j + 1) % 10 == 0:
                        print(f"    已处理 {j + 1}/{len(batch_texts)} 个文本")
                except Exception as e2:
                    print(f"    文本 {j} 处理失败: {e2}")
                    # 使用零向量作为默认值
                    default_embedding = np.zeros(model.get_sentence_embedding_dimension())
                    batch_embeddings.append(default_embedding)
            
            if batch_embeddings:
                batch_embeddings = np.array(batch_embeddings)
                embeddings.append(batch_embeddings)
                print(f"  ✓ 批次 {batch_num} 逐个处理完成")
        
        # 每10个批次显示一次内存使用情况
        if batch_num % 10 == 0:
            try:
                import psutil
                memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
                print(f"  当前内存使用: {memory_usage:.1f} MB")
            except ImportError:
                pass
        
        # 每5个批次显示一次进度
        if batch_num % 5 == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_batch = elapsed_time / batch_num
            remaining_batches = total_batches - batch_num
            estimated_remaining_time = remaining_batches * avg_time_per_batch
            print(f"  进度: {batch_num}/{total_batches} ({batch_num/total_batches*100:.1f}%) - 预计剩余时间: {estimated_remaining_time/60:.1f}分钟")
    
    # 合并所有批次的结果
    print("正在合并所有批次的结果...")
    try:
        all_embeddings = np.vstack(embeddings)
        total_time = time.time() - start_time
        print(f"✓ 嵌入向量生成完成！")
        print(f"✓ 最终向量形状: {all_embeddings.shape}")
        print(f"✓ 总用时: {total_time/60:.1f}分钟")
        print(f"✓ 平均每批次: {total_time/total_batches:.2f}秒")
    except Exception as e:
        print(f"✗ 合并批次失败: {e}")
        return None
    
    return all_embeddings, valid_texts

def save_embeddings(embeddings, output_file='embedding/emb.npy'):
    """保存嵌入向量"""
    print(f"\n=== 保存嵌入向量 ===")
    
    # 确保embedding目录存在
    os.makedirs('embedding', exist_ok=True)
    print(f"确保输出目录存在: embedding/")
    
    try:
        print(f"正在保存到: {output_file}")
        np.save(output_file, embeddings)
        file_size = os.path.getsize(output_file) / (1024*1024)
        print(f"✓ 嵌入向量已保存到: {output_file}")
        print(f"✓ 文件大小: {file_size:.2f} MB")
        return True
    except Exception as e:
        print(f"✗ 保存失败: {e}")
        return False

def analyze_embeddings(embeddings):
    """分析嵌入向量"""
    print(f"\n=== 嵌入向量分析 ===")
    
    print(f"向量形状: {embeddings.shape}")
    print(f"向量维度: {embeddings.shape[1]}")
    print(f"样本数量: {embeddings.shape[0]}")
    
    # 统计信息
    print(f"\n统计信息:")
    print(f"最小值: {embeddings.min():.6f}")
    print(f"最大值: {embeddings.max():.6f}")
    print(f"平均值: {embeddings.mean():.6f}")
    print(f"标准差: {embeddings.std():.6f}")
    
    # 检查是否有NaN或无穷大值
    nan_count = np.isnan(embeddings).sum()
    inf_count = np.isinf(embeddings).sum()
    
    print(f"NaN值数量: {nan_count}")
    print(f"无穷大值数量: {inf_count}")
    
    if nan_count > 0 or inf_count > 0:
        print("⚠️  警告：发现异常值！")
        # 处理异常值
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
        print("✓ 已处理异常值")
    else:
        print("✓ 未发现异常值")
    
    return embeddings

def main():
    """主函数"""
    print("=" * 60)
    print("微博数据嵌入向量生成工具")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # 设置GPU内存分配策略
        torch.cuda.empty_cache()
    
    # 快速模式选择
    print("\n=== 处理模式选择 ===")
    print("1. 快速模式 (推荐) - 使用轻量级模型，处理速度更快")
    print("2. 标准模式 - 使用平衡模型，质量和速度兼顾")
    print("3. 高质量模式 - 使用高质量模型，处理速度较慢")
    
    # 默认使用快速模式
    mode = "fast"  # 可以改为 "balanced" 或 "quality"
    
    model_map = {
        'fast': 'all-MiniLM-L6-v2',
        'balanced': 'paraphrase-multilingual-MiniLM-L12-v2', 
        'quality': 'paraphrase-multilingual-mpnet-base-v2'
    }
    
    selected_model = model_map.get(mode, 'all-MiniLM-L6-v2')
    print(f"选择模式: {mode} -> 模型: {selected_model}")
    
    # 加载模型
    print("\n" + "="*40)
    print("步骤1: 加载模型")
    print("="*40)
    model = load_model(selected_model)
    if model is None:
        return
    
    # 如果使用GPU，将模型移到GPU
    if device.type == 'cuda':
        print("将模型移到GPU...")
        model = model.to(device)
        print("✓ 模型已移到GPU")
    
    # 加载数据
    print("\n" + "="*40)
    print("步骤2: 加载数据")
    print("="*40)
    data = load_data()
    if data is None:
        return
    
    # 准备文本数据
    print("\n准备文本数据...")
    if isinstance(data, pd.DataFrame):
        texts = data['segmented_text'].fillna('').tolist()
        print(f"从DataFrame提取文本，列名: {list(data.columns)}")
    else:
        texts = data
        print("使用文本列表")
    
    # 根据数据量和模式调整批次大小
    if mode == 'fast':
        if len(texts) > 50000:
            batch_size = 128  # 减小批次大小，避免内存问题
        elif len(texts) > 10000:
            batch_size = 64
        else:
            batch_size = 32
    else:
        if len(texts) > 50000:
            batch_size = 64
        elif len(texts) > 10000:
            batch_size = 32
        else:
            batch_size = 16
    
    print(f"根据数据量和模式调整批次大小: {batch_size}")
    print("注意：如果处理速度太慢，可以手动减小批次大小")
    
    # 如果用户想要更小的批次大小，可以在这里修改
    # batch_size = 32  # 取消注释这行来使用更小的批次
    
    # 生成嵌入向量
    print("\n" + "="*40)
    print("步骤3: 生成嵌入向量")
    print("="*40)
    result = generate_embeddings(model, texts, batch_size)
    if result is None:
        return
    
    embeddings, valid_texts = result
    
    # 分析嵌入向量
    print("\n" + "="*40)
    print("步骤4: 分析嵌入向量")
    print("="*40)
    embeddings = analyze_embeddings(embeddings)
    
    # 保存嵌入向量
    print("\n" + "="*40)
    print("步骤5: 保存结果")
    print("="*40)
    success = save_embeddings(embeddings)
    if not success:
        return
    
    # 保存对应的文本
    print("正在保存对应的文本...")
    with open('embedding/embedding_texts.txt', 'w', encoding='utf-8') as f:
        for text in valid_texts:
            f.write(text + '\n')
    print(f"✓ 对应的文本已保存到: embedding/embedding_texts.txt")
    
    # 保存原始文本（用于BERTopic训练）
    print("正在保存原始文本...")
    try:
        # 加载原始文本数据
        with open('data/文本.txt', 'r', encoding='utf-8') as f:
            original_texts = [line.strip() for line in f if line.strip()]
        
        # 确保原始文本数量与有效文本数量一致
        if len(original_texts) >= len(valid_texts):
            original_texts = original_texts[:len(valid_texts)]
            with open('embedding/original_texts.txt', 'w', encoding='utf-8') as f:
                for text in original_texts:
                    f.write(text + '\n')
            print(f"✓ 原始文本已保存到: embedding/original_texts.txt")
        else:
            print("⚠️  警告：原始文本数量不足，跳过保存")
    except Exception as e:
        print(f"⚠️  警告：保存原始文本失败: {e}")
    
    # 生成报告
    print(f"\n" + "="*40)
    print("生成报告")
    print("="*40)
    print(f"原始文本数量: {len(texts):,}")
    print(f"有效文本数量: {len(valid_texts):,}")
    print(f"嵌入向量数量: {embeddings.shape[0]:,}")
    print(f"向量维度: {embeddings.shape[1]}")
    print(f"内存使用: {embeddings.nbytes / (1024*1024):.2f} MB")
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("嵌入向量生成完成！")
    print("=" * 60)
    
    print(f"\n输出文件:")
    print(f"1. embedding/emb.npy - 嵌入向量")
    print(f"2. embedding/embedding_texts.txt - 对应的文本")
    print(f"3. embedding/original_texts.txt - 原始文本（用于BERTopic）")
    print(f"\n下一步: 运行 5_main_weibo.py 进行主题建模")

if __name__ == "__main__":
    # 设置日志文件
    log_file = "data/log/04_generate_embeddings_log.txt"
    
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
        print(f"嵌入向量生成日志已保存到: {log_file}") 