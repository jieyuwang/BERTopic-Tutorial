#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERTopic测试脚本
"""

import numpy as np
from bertopic import BERTopic
import sys
from sklearn.feature_extraction.text import CountVectorizer
import jieba

def jieba_tokenizer(text):
    return jieba.lcut(text)

def test_bertopic():
    """测试BERTopic基本功能"""
    print("=" * 50)
    print("BERTopic功能测试")
    print("=" * 50)
    
    # 检查BERTopic版本
    try:
        import bertopic
        print(f"BERTopic版本: {bertopic.__version__}")
    except Exception as e:
        print(f"无法获取BERTopic版本: {e}")
    
    # 扩充测试数据
    test_texts = [
        "机器学习是人工智能的一个重要分支，它使计算机能够从数据中学习并做出预测",
        "深度学习是机器学习的一个子领域，它使用多层神经网络来模拟人脑的工作方式",
        "自然语言处理是人工智能的一个分支，专门处理和理解人类语言",
        "计算机视觉是人工智能的一个领域，致力于让计算机理解和处理图像和视频",
        "数据挖掘是从大量数据中发现有用信息和模式的过程",
        "监督学习是机器学习的一种方法，使用标记的训练数据来训练模型",
        "无监督学习不需要标记数据，而是从数据中发现隐藏的模式和结构",
        "神经网络是一种模仿生物神经系统的计算模型",
        "卷积神经网络在图像识别和计算机视觉任务中表现优异",
        "循环神经网络特别适合处理序列数据，如文本和时间序列",
        "人工智能正在改变我们的生活方式，包括医疗、金融和交通等领域",
        "大数据分析帮助企业做出更明智的决策",
        "机器翻译是自然语言处理的重要应用之一",
        "自动驾驶汽车依赖于计算机视觉和深度学习技术",
        "推荐系统通过分析用户行为来推荐商品或内容",
        "语音识别技术让计算机能够理解和转录人类的语音",
        "图神经网络用于处理图结构数据，如社交网络和分子结构",
        "迁移学习可以利用已有模型知识解决新问题",
        "强化学习通过奖励机制让智能体学习最优策略",
        "区块链技术为数据安全和透明性提供了新方法"
    ]
    
    print(f"测试文本数量: {len(test_texts)}")
    
    # 自定义分词器
    vectorizer = CountVectorizer(tokenizer=jieba_tokenizer)
    
    # 创建BERTopic模型
    print("\n创建BERTopic模型...")
    try:
        topic_model = BERTopic(
            min_topic_size=5,    # 小数据集使用较小的最小主题大小
            nr_topics=20,        # 限制主题数量为20个
            top_n_words=10,      # 关键词数量
            verbose=True
        )
        print("✓ BERTopic模型创建成功")
    except Exception as e:
        print(f"✗ BERTopic模型创建失败: {e}")
        return False
    
    # 训练模型
    print("\n训练模型...")
    try:
        topics, probs = topic_model.fit_transform(test_texts)
        print("✓ 模型训练成功")
        print(f"发现主题数量: {len(topic_model.get_topics())}")
        
        # 显示主题信息
        topic_info = topic_model.get_topic_info()
        print("\n主题信息:")
        print(topic_info)
        
        return True
        
    except Exception as e:
        print(f"✗ 模型训练失败: {e}")
        return False

if __name__ == "__main__":
    success = test_bertopic()
    if success:
        print("\n✓ BERTopic测试通过！")
    else:
        print("\n✗ BERTopic测试失败！")
        sys.exit(1) 