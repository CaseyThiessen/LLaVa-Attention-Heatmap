from huggingface_hub import snapshot_download

llava_models_path = 'llava-hf/llava-1.5-7b-hf'

snapshot_download(repo_id="llava-hf/llava-1.5-7b-hf", local_dir=llava_models_path)
