import os
import shutil
import random

def split_dataset(input_dir, output_unlabeled_dir, output_labeled_dir, log_file="missing_files.log"):
    # 确保目标文件夹存在
    folders = ['annotation', 'depth', 'mask', 'normal', 'rgb', 'segm']
    for target_dir in [output_unlabeled_dir, output_labeled_dir]:
        for folder in folders:
            os.makedirs(os.path.join(target_dir, folder), exist_ok=True)

    # 获取所有 RGB 文件路径
    rgb_dir = os.path.join(input_dir, 'rgb')
    rgb_files = [f for f in os.listdir(rgb_dir) if f.endswith('.jpeg')]

    # 随机打乱并分割文件名
    random.shuffle(rgb_files)
    mid_point = len(rgb_files) // 2
    unlabeled_files = rgb_files[:mid_point]
    labeled_files = rgb_files[mid_point:]

    # 用于记录缺失文件
    missing_files = []

    def copy_files(files, target_dir):
        nonlocal missing_files  # 记录缺失文件的全局列表
        for file_name in files:
            # 提取公共基名和索引号
            base_name, index = file_name.split('_rgb_')
            index = index.split('.')[0]  # 去掉文件后缀，只保留索引号

            for folder in folders:
                if folder == 'rgb':
                    # RGB 文件直接复制
                    source_file = os.path.join(input_dir, folder, file_name)
                else:
                    # 其他文件夹的文件名拼接方式
                    folder_extension = {
                        'annotation': '.json',
                        'depth': '.npy.gz',
                        'mask': '.npy.gz',
                        'normal': '.npy.gz',
                        'segm': '.npy.gz',
                    }
                    source_file = os.path.join(
                        input_dir,
                        folder,
                        f"{base_name}_{folder}_{index}{folder_extension[folder]}"
                    )

                target_folder = os.path.join(target_dir, folder)
                target_file = os.path.join(target_folder, os.path.basename(source_file))

                # 如果文件存在，则复制；否则记录缺失文件
                if os.path.exists(source_file):
                    shutil.copy2(source_file, target_file)
                else:
                    missing_files.append(source_file)  # 添加到缺失文件列表

    # 复制未标注文件
    copy_files(unlabeled_files, output_unlabeled_dir)

    # 复制标注文件
    copy_files(labeled_files, output_labeled_dir)

    # 保存缺失文件日志
    with open(log_file, "w") as log:
        log.write("Missing files:\n")
        for missing_file in missing_files:
            log.write(f"{missing_file}\n")

    print(f"数据集分割完成！缺失文件已记录到 {log_file}")

def check_data_integrity(input_dir):
    folders = ['rgb', 'annotation', 'depth', 'mask', 'normal', 'segm']
    counts = {folder: len(os.listdir(os.path.join(input_dir, folder))) for folder in folders}
    print("输入目录中每种文件的数量：")
    for folder, count in counts.items():
        print(f"{folder}: {count} files")

# 主函数调用
if __name__ == "__main__":
    input_dir = 'data/FRONT3D/train'
    output_unlabeled_dir = 'data/unlabeled_FRONT3D/train'
    output_labeled_dir = 'data/labeled_FRONT3D/train'

    # 检查数据完整性
    check_data_integrity(input_dir)

    # 分割数据集
    split_dataset(input_dir, output_unlabeled_dir, output_labeled_dir)
