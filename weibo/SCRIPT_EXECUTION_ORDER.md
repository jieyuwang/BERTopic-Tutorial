# 微博主题建模脚本执行顺序

## 主要执行流程

### 1. 环境准备
```bash
python 0_setup_environment.py
```
- 设置环境变量
- 配置并行处理

### 2. 数据预处理
```bash
python 1_data_preprocessing.py
```
- 加载原始数据
- 基础清洗
- 保存到 `data/weibo_clean_data.csv`

### 3. 词汇分析
```bash
python 2_analyze_vocabulary.py
```
- 分析词汇分布
- 生成词汇统计报告

### 4. 分词处理
```bash
python 3_word_segmentation.py
```
- 使用jieba分词
- 保存分词结果到 `data/weibo_segmented_data.csv`

### 5. 生成嵌入向量
```bash
python 4_generate_embeddings.py
```
- 加载Sentence Transformer模型
- 生成文本嵌入向量
- 保存到 `embedding/emb.npy`
- 保存原始文本到 `embedding/original_texts.txt`

### 6. 主题建模（三个版本）

#### 6.1 标准版本
```bash
python 5_1_main_weibo.py
```
- 使用标准参数
- 结果保存到 `data/` 目录
- 日志：`data/log/05_1_main_weibo_log.txt`

#### 6.2 高质量版本
```bash
python 5_2_improve_bertopic_quality.py
```
- 使用优化参数
- 更好的文本清洗
- 结果保存到 `data/high_quality_results/` 目录
- 日志：`data/log/05_2_improve_bertopic_quality_log.txt`

#### 6.3 修复版本（推荐）
```bash
python 5_3_bertopic_fixed.py
```
- 解决主题质量问题
- 改进中文分词和文本清洗
- 添加完整可视化功能
- 结果保存到 `data/fixed_results/` 目录
- 日志：`data/log/05_3_bertopic_fixed_log.txt`

### 7. 评估指标
```bash
python 6_evaluation_metrics.py
```
- 计算主题质量指标
- 生成评估报告

### 8. 可视化
```bash
python create_visualizations.py
```
- 创建主题分布图
- 生成可视化报告

## 辅助脚本

### 测试脚本
- `test_bertopic.py` - 基础BERTopic测试
- `test_bertopic_fast.py` - 快速测试
- `test_bertopic_simple.py` - 简化测试
- `test_embedding_speed.py` - 嵌入速度测试

### 工具脚本
- `create_original_texts.py` - 生成原始文本文件
- `load_and_use_model.py` - 加载和使用已保存的模型

### 一键执行
```bash
python 00_run_pipeline.py
```
- 自动执行完整流程（1-5步）

## 输出文件结构

```
data/
├── weibo_clean_data.csv              # 清洗后的数据
├── weibo_segmented_data.csv          # 分词后的数据
├── log/                              # 日志文件
│   ├── 01_preprocessing_log.txt
│   ├── 02_vocabulary_analysis_log.txt
│   ├── 03_word_segmentation_log.txt
│   ├── 04_generate_embeddings_log.txt
│   ├── 05_1_main_weibo_log.txt
│   ├── 05_2_improve_bertopic_quality_log.txt
│   └── 05_3_bertopic_fixed_log.txt
└── evaluation_results.csv            # 评估结果

embedding/
├── emb.npy                           # 嵌入向量
└── original_texts.txt                # 原始文本

results/                              # 主题建模结果
├── standard_results/                 # 标准版本结果
│   ├── topic_modeling_results.csv
│   ├── topic_info.csv
│   └── bertopic_model/
├── high_quality_results/             # 高质量版本结果
│   ├── topic_modeling_results_high_quality.csv
│   ├── topic_info_high_quality.csv
│   └── bertopic_model_high_quality/
└── fixed_results/                    # 修复版本结果
    ├── topic_modeling_results_fixed.csv
    ├── topic_info_fixed.csv
    ├── bertopic_model_fixed/
    └── visualizations/               # 可视化图表
        ├── topic_distribution.png
        ├── topic_keywords_heatmap.png
        ├── document_distribution.png
        ├── topic_probability_distribution.png
        └── fixed_topic_analysis_report.txt
```

## 注意事项

1. **执行顺序**：建议按顺序执行，确保数据依赖关系
2. **内存需求**：嵌入生成步骤需要较大内存
3. **时间消耗**：完整流程可能需要1-2小时
4. **版本选择**：可以选择标准版本或高质量版本，或两者都运行
5. **日志查看**：所有步骤的详细日志都保存在 `data/log/` 目录 