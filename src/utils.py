"""配置与数据IO"""

import yaml
import pickle
import os


def load_config(path):
    """加载YAML配置文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_data(path):
    """从pickle加载数据字典"""
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_data(datas, path):
    """保存数据字典到pickle"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(datas, f)
