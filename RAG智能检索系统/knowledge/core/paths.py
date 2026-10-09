import os

from pathlib import Path
KNOWLEDGE_ROOT = Path(__file__).resolve().parent.parent
LOCAL_BASE_DIR = KNOWLEDGE_ROOT/'temp_data'
FRONT_PAGE_DIR = KNOWLEDGE_ROOT/'front'


def get_local_base_dir():
    """获取本地文件存储基础目录"""
    return LOCAL_BASE_DIR

def get_front_page_dir():
    """获取前端静态页面目录"""
    return  FRONT_PAGE_DIR