import numpy as np
import os

# 指定文件夹路径
folder_path = 'ssr/depth_anything/'

# 遍历文件夹下的所有 .npy 文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.npy'):
        file_path = os.path.join(folder_path, file_name)
        
        # 加载 .npy 文件
        data = np.load(file_path)
        
        # 检查数据形状，如果是三通道则处理
        if data.shape[-1] == 3:
            # 假设三个通道的数据相同，取第一个通道
            single_channel_data = data[:, :, 0]
            
            # 保存为原文件名，覆盖原始文件
            np.save(file_path, single_channel_data)
            print(f"Converted {file_name} to single channel.")
        else:
            print(f"{file_name} is not a 3-channel file, skipping.")

print("Conversion completed.")
