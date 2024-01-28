from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen-1_8B-Chat', cache_dir='../download_llm')
print("模型下载完成")