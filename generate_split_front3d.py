import os
import json

# 图片和标注文件路径
rgb_dir = "/home/data/luoxi/SSR-code/data/labeled_FRONT3D/val/rgb"
annotation_dir = "/home/data/luoxi/SSR-code/data/labeled_FRONT3D/val/annotation"
output_split_dir = "/home/data/luoxi/SSR-code/dataset/labeled_FRONT3D/split/val"

# 分类名称映射表
classnames = {
    "bed": "bed",
    "bookshelf": "bookshelf",
    "shelf": "bookshelf",
    "cabinet": "cabinet",
    "chair": "chair",
    "desk": "desk",
    "nightstand": "night_stand",
    "night_stand": "night_stand",
    "sofa": "sofa",
    "table": "table",
    "dresser": "dresser",
    "dressing table": "dresser",  # Dressing Table 全名匹配到 dresser
    "dressingtable": "dresser"
}

# 确保输出文件夹存在
os.makedirs(output_split_dir, exist_ok=True)

# 初始化类别文件
split_data = {classname: [] for classname in classnames.values()}

# 遍历 annotation 文件夹
for annotation_file in os.listdir(annotation_dir):
    annotation_path = os.path.join(annotation_dir, annotation_file)

    # 解析 JSON 文件
    with open(annotation_path, "r") as f:
        annotation_data = json.load(f)

    # 获取对应的 RGB 文件名
    base_filename = annotation_file.replace("_annotation_", "_rgb_").replace(".json", ".jpeg")
    rgb_path = os.path.join("train/rgb", base_filename)

    # 遍历物体字典
    for obj_id, obj_data in annotation_data["obj_dict"].items():
        # 提取类别标签
        full_label = obj_data["label"][0].lower()  # 转为小写以避免大小写匹配问题

        # 1. 全名匹配
        if full_label in classnames:
            target_class = classnames[full_label]
            split_data[target_class].append([rgb_path, obj_id, target_class])
        else:
            # 2. 如果全名匹配失败，提取单词的最后一个词进行匹配
            last_word_label = full_label.split()[-1]
            if last_word_label in classnames:
                target_class = classnames[last_word_label]
                split_data[target_class].append([rgb_path, obj_id, target_class])

# 将结果写入对应的 JSON 文件
for classname, data in split_data.items():
    output_file = os.path.join(output_split_dir, f"{classname}.json")
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

print(f"JSON files have been generated in {output_split_dir}")
