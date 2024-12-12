import torch

# 加载模型文件
model_path = 'output/front3d_ckpt/model_latest.pth'
checkpoint = torch.load(model_path)

# 获取 net 的 state_dict
net_state_dict = checkpoint['net']

# 输出原始参数值并修改
if 'module.dino_weight' in net_state_dict:
    print('Original dino_weight:', net_state_dict['module.dino_weight'].item())  # 输出原值
    net_state_dict['module.dino_weight'] = torch.nn.Parameter(torch.tensor(0.12))  # 修改值
    print('Modified dino_weight:', net_state_dict['module.dino_weight'].item())   # 输出新值

if 'module.encoder.diffu_weight' in net_state_dict:
    print('Original diffu_weight:', net_state_dict['module.encoder.diffu_weight'].item())  # 输出原值
    net_state_dict['module.encoder.diffu_weight'] = torch.nn.Parameter(torch.tensor(0.85))  # 修改值
    print('Modified diffu_weight:', net_state_dict['module.encoder.diffu_weight'].item())   # 输出新值

# 保存修改后的模型
torch.save(checkpoint, model_path)
print('参数已修改并保存成功。')