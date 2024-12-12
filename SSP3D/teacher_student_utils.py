import torch

def update_teacher_model(teacher_model, student_model, alpha):
    """
    更新教师模型权重 (EMA 更新).
    Args:
        teacher_model (torch.nn.Module): 教师模型.
        student_model (torch.nn.Module): 学生模型.
        alpha (float): EMA 衰减系数.
    """
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data = alpha * teacher_param.data + (1 - alpha) * student_param.data


def filter_pseudo_labels(teacher_outputs, threshold=0.6):
    """
    根据教师模型输出过滤伪标签.
    Args:
        teacher_outputs (dict): 教师模型的输出，包含伪标签.
        threshold (float): 过滤阈值.
    Returns:
        filtered_outputs (dict): 可信度高的伪标签.
    """
    filtered_outputs = {}
    for key, value in teacher_outputs.items():
        confidence = value['confidence']  # 假设输出包含 confidence 字段
        mask = confidence >= threshold
        filtered_outputs[key] = value[mask]
    return filtered_outputs
