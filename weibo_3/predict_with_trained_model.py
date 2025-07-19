import pandas as pd
import pickle
import os
from weibo_3.behavior_classification_with_bertopic import BERTopicBehaviorClassifier, setup_logger

def predict_new_data(input_xlsx, model_pkl, output_xlsx, text_col='txt'):
    logger = setup_logger()
    logger.info(f"加载模型: {model_pkl}")
    with open(model_pkl, 'rb') as f:
        model_data = pickle.load(f)
    
    classifier = BERTopicBehaviorClassifier()
    classifier.topic_model = model_data['topic_model']
    classifier.embedding_model = model_data['embedding_model']
    classifier.classifier = model_data['classifier']
    classifier.vectorizer = model_data['vectorizer']
    classifier.topic_behavior_mapping = model_data['topic_behavior_mapping']
    classifier.behavior_keywords = model_data['behavior_keywords']

    logger.info(f"读取数据: {input_xlsx}")
    df = pd.read_excel(input_xlsx)
    if text_col not in df.columns:
        raise ValueError(f"输入文件缺少'{text_col}'列")
    
    logger.info("文本预处理...")
    df['processed_text'] = df[text_col].apply(classifier.preprocess_text)
    texts = df['processed_text'].tolist()

    logger.info("预测行为类别...")
    preds, probs = classifier.predict_behaviors(texts)
    df['predicted_behavior'] = preds
    
    # 可选：输出每类概率
    if probs is not None:
        for i, cls in enumerate(classifier.classifier.classes_):
            df[f'prob_{cls}'] = probs[:, i]

    logger.info(f"保存预测结果到: {output_xlsx}")
    df.to_excel(output_xlsx, index=False)
    logger.info("预测完成！")

if __name__ == '__main__':
    input_xlsx = '../weibo_3/data/data_new.xlsx'
    model_pkl = '../weibo_3/result/behavior_analysis/bertopic_behavior_model-old.pkl'
    output_xlsx = '../weibo_3/data/data_new_with_pred_old.xlsx'
    predict_new_data(input_xlsx, model_pkl, output_xlsx) 