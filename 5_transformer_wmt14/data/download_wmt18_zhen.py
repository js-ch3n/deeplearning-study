"""
下载 WMT18 zh-en 数据集到本地缓存。
运行: python data/download_wmt18_zhen.py
"""
from datasets import load_dataset

cache_dir = "../data/hf_cache"

print("下载 wmt18 zh-en (train)...")
load_dataset("wmt/wmt18", "zh-en", split="train", cache_dir=cache_dir)

print("下载 wmt18 zh-en (validation)...")
load_dataset("wmt/wmt18", "zh-en", split="validation", cache_dir=cache_dir)

print("下载 wmt18 zh-en (test)...")
load_dataset("wmt/wmt18", "zh-en", split="test", cache_dir=cache_dir)

print("完成！缓存目录:", cache_dir)
