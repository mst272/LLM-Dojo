from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen1.5-0.5B', cache_dir='../../../download_llm')
print("模型下载完成")