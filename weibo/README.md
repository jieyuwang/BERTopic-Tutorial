# 微博数据BERTopic主题建模项目

## 📖 项目概述

本项目使用BERTopic对微博数据进行主题建模，包含完整的数据预处理、嵌入生成、主题建模和评估流程。

## 🚀 快速开始

### 一键运行完整流程
```bash
# 1. 确保数据文件存在
ls data/data_new.xlsx

# 2. 运行完整流程
python 00_run_pipeline.py
```

### 环境设置
```bash
# 自动设置环境
python 0_setup_environment.py

# 或使用conda
conda env create -f requirements_minimal.yml
conda activate weibo-bertopic-minimal
```

## 📁 项目结构

```
weibo/
├── 00_run_pipeline.py              # 完整流程运行脚本
├── 0_setup_environment.py          # 环境设置脚本
├── 1_data_preprocessing.py         # 数据预处理脚本
├── 2_analyze_vocabulary.py         # 词汇分析脚本
├── 3_word_segmentation.py          # 分词处理脚本
├── 4_generate_embeddings.py        # 嵌入向量生成脚本
├── 5_main_weibo.py                 # 主题建模主程序
├── 6_evaluation_metrics.py         # 评估指标计算脚本
├── data/                           # 数据目录
│   ├── data_new.xlsx              # 原始微博数据
│   ├── 文本.txt                   # 清理后的文本（1_data_preprocessing.py产出）
│   ├── 时间.txt                   # 时间信息（1_data_preprocessing.py产出）
│   ├── weibo_clean_data.csv       # 清理后的完整数据（1_data_preprocessing.py产出）
│   ├── 切词.txt                   # 分词结果（3_word_segmentation.py产出）
│   └── embedding_*.npy            # 嵌入向量文件（4_generate_embeddings.py产出）
├── results/                        # 结果输出目录
│   ├── weibo_complete_results.csv # 完整聚类结果
│   ├── weibo_topic_info.csv       # 主题详细信息
│   ├── weibo_analysis_stats.csv   # 分析统计
│   └── weibo_evaluation_metrics.csv # 评估指标
├── vocabulary_analysis/            # 词汇分析结果
│   └── wordcloud.png              # 词汇云图
├── embedding/                      # 嵌入向量文件
├── 分词/                          # 分词相关文件
│   ├── stopwords.txt              # 停用词表
│   ├── userdict.txt               # 用户词典
│   └── userdict_updated.txt       # 更新后的用户词典
├── requirements.yml               # 完整环境配置
├── requirements_minimal.yml       # 最小环境配置
└── README.md                      # 项目说明文档
```

## 🔄 处理流程

### 完整流程步骤
1. **数据预处理** (`1_data_preprocessing.py`) - 清理Excel数据，提取文本和时间信息
   - 产出：`data/文本.txt`, `data/时间.txt`, `data/weibo_clean_data.csv`
2. **词汇分析** (`2_analyze_vocabulary.py`) - 统计高频词汇，生成用户自定义词典建议
   - 产出：`vocabulary_analysis/wordcloud.png`, `分词/userdict_updated.txt`
3. **分词处理** (`3_word_segmentation.py`) - 使用jieba进行中文分词
   - 产出：`data/切词.txt`, `data/切词_word_freq.csv`
4. **嵌入向量生成** (`4_generate_embeddings.py`) - 生成文本嵌入向量
   - 产出：`data/embedding_sen.npy`, `data/embedding_bert.npy`, `data/embedding_info.csv`
5. **主题建模** (`5_main_weibo.py`) - 使用BERTopic进行主题建模
   - 产出：`results/weibo_*.csv` 系列文件
6. **评估分析** (`6_evaluation_metrics.py`) - 计算主题建模评估指标
   - 产出：`results/weibo_evaluation_metrics.csv`

### 单独运行步骤
```bash
# 环境设置
python 0_setup_environment.py

# 数据预处理
python 1_data_preprocessing.py

# 词汇分析
python 2_analyze_vocabulary.py

# 分词处理
python 3_word_segmentation.py

# 嵌入向量生成
python 4_generate_embeddings.py

# 主题建模
python 5_main_weibo.py

# 评估分析
python 6_evaluation_metrics.py
```

## 📊 输入输出文件

### 输入文件
- `data/data_new.xlsx` - 原始微博数据
  - 应包含列：`txt`(文本), `date`(时间), `user`(用户), `repost`(转发), `comment`(评论), `like`(点赞), `type`(类型)

### 主要输出文件
- `results/weibo_complete_results.csv` - 完整聚类结果（包含原文、时间、主题标签）
- `results/weibo_topic_info.csv` - 主题详细信息（关键词、代表性文档）
- `results/weibo_analysis_stats.csv` - 分析统计（文档数、主题数、覆盖率）
- `results/weibo_evaluation_metrics.csv` - 评估指标（一致性、多样性、聚类质量）

### 词汇分析结果
- `vocabulary_analysis/wordcloud.png` - 词汇云图
- `分词/userdict_updated.txt` - 更新后的用户词典

### 中间文件
- `data/文本.txt` - 清理后的文本（1_data_preprocessing.py产出）
- `data/时间.txt` - 时间信息（1_data_preprocessing.py产出）
- `data/weibo_clean_data.csv` - 清理后的完整数据（1_data_preprocessing.py产出）
- `data/切词.txt` - 分词结果（3_word_segmentation.py产出）
- `data/切词_word_freq.csv` - 词频统计（3_word_segmentation.py产出）
- `data/embedding_*.npy` - 嵌入向量文件（4_generate_embeddings.py产出）
- `data/embedding_info.csv` - 嵌入向量信息（4_generate_embeddings.py产出）

## ⚙️ 环境配置

### 完整环境（推荐）
```bash
conda env create -f requirements.yml
conda activate BERTopic-Tutorial
```

### 最小环境（快速测试）
```bash
conda env create -f requirements_minimal.yml
conda activate BERTopic-Tutorial-Minimal
```

## 🔧 参数配置

### 聚类参数调整
在`5_main_weibo.py`中可以调整：
```python
# HDBSCAN聚类参数
hdbscan_model = HDBSCAN(
    min_cluster_size=20,  # 最小聚类大小
    min_samples=10,       # 最小样本数
    metric='euclidean'    # 距离度量
)

# UMAP降维参数
umap_model = UMAP(
    n_neighbors=15,       # 邻居数
    min_dist=0.0,         # 最小距离
    metric='cosine'       # 距离度量
)
```

### 文本处理参数
在`3_word_segmentation.py`中可以调整：
```python
def segment_text(text, stopwords, min_length=2):  # 最小词长度
```

### 嵌入模型选择
在`4_generate_embeddings.py`中可以更换模型：
```python
# Sentence-Transformers模型
model_name = 'paraphrase-multilingual-MiniLM-L12-v2'

# BERT模型
model_name = 'bert-base-chinese'
```

## 📈 性能优化

### 内存优化
- 减小批处理大小：`batch_size = 32`
- 使用更小的嵌入模型
- 分批处理大数据集

### 计算优化
- 使用GPU加速（如果可用）
- 调整UMAP的`low_memory=True`参数
- 使用更轻量级的嵌入模型

### 并行处理
- 调整批处理大小
- 使用多进程处理
- 优化内存使用

## 🐛 常见问题

### 1. 内存不足
- 减少批处理大小
- 使用更小的模型
- 增加系统内存

### 2. 依赖冲突
- 使用conda环境隔离
- 检查包版本兼容性
- 重新安装冲突的包

### 3. 中文显示问题
- 安装中文字体
- 设置matplotlib字体
- 使用支持中文的字体

### 4. 模型下载慢
- 使用国内镜像源
- 手动下载模型文件
- 使用代理

### 5. 聚类效果不佳
- 调整`min_cluster_size`参数
- 增加`min_samples`值
- 尝试不同的距离度量

### 6. 分词效果不理想
- 更新用户自定义词典
- 调整停用词列表
- 检查文本预处理质量

## 💡 使用建议

1. **首次运行**：建议先运行词汇分析，查看并调整用户词典
2. **大数据集**：嵌入向量生成可能需要较长时间，请耐心等待
3. **内存优化**：如果内存不足，可以调整批处理大小
4. **结果分析**：查看主题信息，根据需要调整BERTopic参数
5. **词汇优化**：根据词汇分析结果，优化用户自定义词典

## 📞 技术支持

如遇到问题，请检查：
1. Python版本（推荐3.10）
2. 依赖包版本兼容性
3. 系统内存和存储空间
4. 网络连接（模型下载）
5. 数据文件格式和内容

## 📝 更新日志

- v1.0.0: 初始版本，包含完整流程
- v1.1.0: 添加词汇分析功能
- v1.2.0: 优化环境配置和错误处理
- v1.3.0: 重新组织脚本结构，按执行顺序编号
- v1.4.0: 完善文档和评估指标

## 📝 许可证

本项目仅供学习和研究使用。 