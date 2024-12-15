import torch
import math
from collections import OrderedDict

def ema_update(teacher_model, student_model, step, total_steps, initial_momentum, learning_rate=1):
    """
    SSP3D 风格的教师模型更新 (EMA)
    
    Args:
        teacher_model: 教师模型
        student_model: 学生模型
        step: 当前时间步
        total_steps: 总时间步
        initial_momentum: 初始动量值
    """
    # 余弦退火调整动量
    momentum = 1 - (1 - initial_momentum) * (math.cos(math.pi * step / total_steps) + 1) / 2
    # 正弦退火调整动量
    # momentum = initial_momentum + (1 - initial_momentum) * (math.sin(math.pi * step / total_steps) ** 2)

    # 教师模型参数更新
    teacher_state = teacher_model.state_dict()
    student_state = student_model.state_dict()
    

    # 去掉学生模型中的 'module.' 前缀，动态调整键名
    student_state_no_prefix = {k[len("module."):] if k.startswith("module.") else k: v for k, v in student_state.items()}

    # 计算学生模型的梯度
    student_gradients = {k: v.grad for k, v in student_model.named_parameters() if v.grad is not None}

    new_teacher_state = OrderedDict()
    for key, value in teacher_state.items():
        # if key in student_state_no_prefix:
        #     # 更新教师模型参数
        #     new_teacher_state[key] = student_state_no_prefix[key] * (1 - momentum) + value * momentum
        if key in student_state_no_prefix:
            # 梯度方向调整权重
            if key in student_gradients:
                # 获取梯度方向并调整教师模型权重
                gradient_direction = student_gradients[key].sign()  # 梯度方向
                gradient_adjustment = -learning_rate * gradient_direction  # 根据学习率调整权重
                adjusted_weight =  gradient_adjustment  # 根据梯度调整权重 (方案 2: 快速适应任务目标)
                print(value)
                # adjusted_weight = value + gradient_adjustment  # 根据梯度调整权重 (方案 1: 平滑更新 (注释中备用))
            else:
                adjusted_weight = value  # 如果没有梯度信息，保持原值

            # 融合 EMA 和梯度方向调整 
            new_teacher_state[key] = adjusted_weight * (1 - momentum) + value * momentum
        else:
            raise ValueError(f"Key {key} not found in student model state dict (after prefix adjustment)")
        
        

    # 加载更新后的教师模型状态
    teacher_model.load_state_dict(new_teacher_state)
