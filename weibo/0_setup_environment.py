#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
微博BERTopic项目环境设置脚本
自动检测并安装所需的依赖包
"""

import subprocess
import sys
import os
import importlib

def check_package(package_name):
    """检查包是否已安装"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def install_package(package_name, pip_name=None):
    """安装包"""
    if pip_name is None:
        pip_name = package_name
    
    print(f"正在安装 {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        print(f"✅ {package_name} 安装成功")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ {package_name} 安装失败")
        return False

def main():
    """主函数"""
    print("=== 微博BERTopic项目环境设置 ===")
    
    # 核心依赖包列表
    core_packages = [
        ("pandas", "pandas>=2.0.0"),
        ("numpy", "numpy>=1.24.0"),
        ("sklearn", "scikit-learn>=1.3.0"),
        ("matplotlib", "matplotlib>=3.7.0"),
        ("seaborn", "seaborn>=0.12.0"),
        ("jieba", "jieba>=0.42.1"),
        ("wordcloud", "wordcloud>=1.9.0"),
        ("bertopic", "bertopic>=0.15.0"),
        ("sentence_transformers", "sentence-transformers>=2.2.0"),
        ("transformers", "transformers>=4.30.0"),
        ("torch", "torch>=2.0.0"),
        ("hdbscan", "hdbscan>=0.8.29"),
        ("umap", "umap-learn>=0.5.3"),
        ("openpyxl", "openpyxl>=3.1.0"),
        ("tqdm", "tqdm>=4.65.0"),
    ]
    
    # 可选依赖包列表
    optional_packages = [
        ("plotly", "plotly>=5.15.0"),
        ("bokeh", "bokeh>=3.2.0"),
        ("scipy", "scipy>=1.10.0"),
        ("joblib", "joblib>=1.3.0"),
        ("psutil", "psutil>=5.9.0"),
        ("pypinyin", "pypinyin>=0.49.0"),
        ("zhon", "zhon>=1.1.5"),
        ("pyLDAvis", "pyLDAvis>=3.4.0"),
        ("gensim", "gensim>=4.3.0"),
        ("rich", "rich>=13.0.0"),
        ("pyyaml", "pyyaml>=6.0"),
        ("loguru", "loguru>=0.7.0"),
    ]
    
    print("\n检查核心依赖包...")
    missing_core = []
    for package, pip_name in core_packages:
        if check_package(package):
            print(f"✅ {package} 已安装")
        else:
            print(f"❌ {package} 未安装")
            missing_core.append((package, pip_name))
    
    print("\n检查可选依赖包...")
    missing_optional = []
    for package, pip_name in optional_packages:
        if check_package(package):
            print(f"✅ {package} 已安装")
        else:
            print(f"⚠️  {package} 未安装 (可选)")
            missing_optional.append((package, pip_name))
    
    # 安装缺失的核心包
    if missing_core:
        print(f"\n需要安装 {len(missing_core)} 个核心依赖包:")
        for package, pip_name in missing_core:
            print(f"  - {package}")
        
        response = input("\n是否安装这些核心依赖包? (y/n): ").lower().strip()
        if response in ['y', 'yes', '是']:
            success_count = 0
            for package, pip_name in missing_core:
                if install_package(package, pip_name):
                    success_count += 1
            
            print(f"\n核心依赖包安装完成: {success_count}/{len(missing_core)} 成功")
        else:
            print("跳过核心依赖包安装")
    else:
        print("\n✅ 所有核心依赖包已安装")
    
    # 安装缺失的可选包
    if missing_optional:
        print(f"\n发现 {len(missing_optional)} 个可选依赖包:")
        for package, pip_name in missing_optional:
            print(f"  - {package}")
        
        response = input("\n是否安装这些可选依赖包? (y/n): ").lower().strip()
        if response in ['y', 'yes', '是']:
            success_count = 0
            for package, pip_name in missing_optional:
                if install_package(package, pip_name):
                    success_count += 1
            
            print(f"\n可选依赖包安装完成: {success_count}/{len(missing_optional)} 成功")
        else:
            print("跳过可选依赖包安装")
    
    # 创建必要目录
    print("\n创建项目目录结构...")
    directories = [
        "data",
        "results", 
        "vocabulary_analysis",
        "embedding",
        "分词"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 {directory}")
    
    print("\n=== 环境设置完成 ===")
    print("\n下一步:")
    print("1. 将微博数据文件放入 data/ 目录")
    print("2. 运行 python run_pipeline.py 开始完整流程")
    print("3. 或运行单个脚本进行特定步骤")

if __name__ == "__main__":
    main() 