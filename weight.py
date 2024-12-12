import torch

# 加载模型
model_path = "/home/data/luoxi/SSR-code/output/front3d_ckpt/model_latest.pth"
checkpoint = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# 打印顶层 keys
print("Top-level Checkpoint keys: ", checkpoint.keys())

# 递归搜索指定 key（如 feature_weights）
def search_key(d, target_key):
    if isinstance(d, dict):
        for key, value in d.items():
            if key == target_key:
                print(f"Found '{target_key}': {value}")
                return value  # 找到后返回值
            elif isinstance(value, dict):
                result = search_key(value, target_key)  # 递归搜索
                if result is not None:
                    return result  # 如果找到了就返回，不再继续搜索
    return None

# 搜索 'feature_weights'
feature_weights = search_key(checkpoint, 'module.encoder.feature_weights')

if feature_weights is None:
    print("'feature_weights' not found in the checkpoint.")
else:
    print("Feature Weights successfully found and loaded!")
