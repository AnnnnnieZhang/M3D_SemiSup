import os
import time
import datetime
import itertools

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.rend_util import get_psnr
from SSP3D.EMA_update import ema_update
from SSP3D.discriminator import Discriminator





def resize_sdf(batch_size,num_points_per_dim,sdf):
    # 假设 real_sdf 和 fake_sdf 是 (batch_size * num_points, 1)

    num_points = sdf.shape[0]//batch_size

    sdf = sdf.view(batch_size, num_points, 1)

    # 假设点的 3D 坐标为以下形式 (batch_size, num_points, 3)
    coords = torch.randint(0, num_points_per_dim, (batch_size, num_points, 3)).to(sdf.device)  # 示例随机坐标

    # 创建规则网格
    sdf_grid = torch.zeros((batch_size, 1, num_points_per_dim, num_points_per_dim, num_points_per_dim)).to(sdf.device)

    # 将点采样数据填充到规则网格
    for b in range(batch_size):
        sdf_grid[b, 0, coords[b, :, 0], coords[b, :, 1], coords[b, :, 2]] = sdf[b, :, 0]

    return sdf_grid




def Recon_trainer(cfg,model,loss,optimizer,scheduler,train_loader,unlabel_train_loader,test_loader,device,checkpoint,tea_net):
    
    start_t = time.time()
    config = cfg.config

    log_dir = cfg.save_path
    os.makedirs(log_dir, exist_ok=True)

    cfg.write_config()
    tb_logger = SummaryWriter(log_dir)


    # 加载判别器
    discriminator = Discriminator().to(device)
    disc_checkpoint_path = config["dict_checkpoint_weight"]
    if os.path.exists(disc_checkpoint_path):
        discriminator.load_state_dict(torch.load(disc_checkpoint_path, map_location=device))
        cfg.log_string(f"Discriminator model loaded from {disc_checkpoint_path}")
    else:
        raise FileNotFoundError(f"Discriminator checkpoint not found at {disc_checkpoint_path}")

    discriminator.eval()  # 判别器设置为评估模式
    filt_val = 0.8



    start_epoch = 0
    iter = 0
    if config["resume"] == True:
        checkpoint.load(config["weight"])
        # start_epoch = scheduler.last_epoch
        start_epoch = checkpoint.module_dict['epoch']
        iter = checkpoint.module_dict['iter']
    if config['finetune']==True:
        start_epoch=0
    scheduler.last_epoch = start_epoch

    model.train()

    min_eval_loss = 10000
    min_eval_loss_teacher = 10000  # 初始化教师模型最优验证损失

    accumulation_steps = config["data"]["accumulation_steps"]  # 累积步数，尝试不同的值

    min_len = min(len(train_loader), len(unlabel_train_loader))
    
    
    for e in range(start_epoch, config['other']['nepoch']):
        torch.cuda.empty_cache()
        cfg.log_string("Switch Phase to Train")
        model.train()
        optimizer.zero_grad()   # 重置优化器梯度
        total_prob = 0
        total_unlabel = 0
        for batch_id, ((indices, model_input, ground_truth), (unlabel_indices, unlabel_model_input)) in enumerate(
            itertools.islice(zip(train_loader, unlabel_train_loader), min_len)
        ):
            unlabel_batch = config['data']['batch_size']['unlabel']
            '''
            indices: [B*1]
            model_input: 
                1.image: [B, 3, H, W]
                2.uv: [B, H*W, 2]                           image coordinate(pixel)
                3.intrinsics: [B, 3, 3]                     image to camera
                4.pose: [B, 4, 4]                           camera to world
                5*.add_points_world: [B, N_uv, N_add, 3]     add points in world coords
            ground_truth:
                1.rgb: [B, num_pixels, 3]
                2.depth: [B, num_pixels, 1]
                3.normal: [B, num_pixels, 3]
            '''

            # 确保所有数据在 GPU 上
            model_input = {k: v.cuda().to(torch.float32) for k, v in model_input.items()}
            unlabel_model_input = {k: v.cuda().to(torch.float32) for k, v in unlabel_model_input.items()}
            ground_truth = {k: v.cuda().to(torch.float32) if isinstance(v, torch.Tensor) else v for k, v in ground_truth.items()}

            tea_net = tea_net.to(device)
                
            # 推理模式下运行教师模型
            
            teacher_outputs = tea_net(unlabel_model_input, unlabel_indices)
            # 过滤伪标签 ……
            # 获得伪标签的 SDF
            num_points_per_dim = 128  # 将目标 3D 网格的边长降低
            tea_sdf = teacher_outputs['sdf'].reshape(-1, 1)
            tea_sdf_grid = resize_sdf(unlabel_batch,num_points_per_dim,tea_sdf)

            # 使用判别器预测伪标签质量
            with torch.no_grad():
                prob = discriminator(tea_sdf_grid)
                mean_prob = prob.mean().item()
                total_prob += mean_prob

            # 根据判别器输出过滤伪标签
            if mean_prob < filt_val:
                unlabel_batch = 0
            
            total_unlabel+=unlabel_batch

            cfg.log_string(f"Batch {batch_id}: Discriminator probability = {mean_prob}, unlabel_batch = {unlabel_batch}")
            
            # 学生模型监督学习
            model_outputs = model(model_input, indices)
            supervised_loss = loss(model_outputs, ground_truth, e, label=True)

            # 学生模型无监督学习（伪标签损失）
            if unlabel_batch == 0:
                pseudo_loss = {key: 0 for key in supervised_loss.keys()}  # 结构一致，值为0
            else:
                pseudo_model_outputs = model(unlabel_model_input, unlabel_indices)
                pseudo_loss = loss(pseudo_model_outputs, teacher_outputs, e, label=False)

            # 加权融合损失字典
            loss_weight_pseudo = unlabel_batch / (config['data']['batch_size']['train'] + unlabel_batch)
            loss_output = {
                key: (1 - loss_weight_pseudo) * supervised_loss[key]
                    + loss_weight_pseudo * pseudo_loss[key]
                for key in supervised_loss.keys()
            }

            # 提取总损失
            total_loss = loss_output['total_loss']

            # 梯度累积
            # 缩放后的损失值用于反向传播和梯度累积
            scaled_total_loss = total_loss / accumulation_steps
            scaled_total_loss.backward()    
            # total_loss.backward()

            total_norm = 0
            '''gradient clip'''
            if (batch_id + 1) % accumulation_steps == 0:
                # print("Student model state dict keys:", model.state_dict().keys())
                # print("Teacher model state dict keys:", tea_net.state_dict().keys())
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=0.7, norm_type=2)
                
                # 计算梯度范数
                total_norm = 0
                parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                optimizer.step()
                optimizer.zero_grad()
            

            psnr = get_psnr(model_outputs['rgb_values'], ground_truth['rgb'].cuda().reshape(-1,3))
            msg = '{:0>8},[epoch {}] ({}/{}): total_loss = {}, rgb_loss = {}, eikonal_loss = {}, depth_loss = {}, normal_l1 = {}, normal_cos = {}, ray_mask_loss = {}, instance_mask_loss = {}, sdf_loss = {}, vis_sdf_loss = {}, psnr = {}, bete={}, alpha={} '.format(
                    str(datetime.timedelta(seconds=round(time.time() - start_t))),
                    e,
                    batch_id + 1,
                    min_len,
                    total_loss.item(),
                    loss_output['rgb_loss'].item(),
                    loss_output['eikonal_loss'].item(),
                    loss_output['depth_loss'].item(),
                    loss_output['normal_l1'].item(),
                    loss_output['normal_cos'].item(),
                    loss_output['ray_mask_loss'].item(),
                    loss_output['instance_mask_loss'].item(),
                    loss_output['sdf_loss'].item(),
                    loss_output['vis_sdf_loss'].item(),
                    psnr.item(),
                    model.module.density.get_beta().item(),
                    1. / model.module.density.get_beta().item(),
                    # model_outputs['depth_weight'].item()
                )
            cfg.log_string(msg)


            tb_logger.add_scalar('Loss/total_loss', total_loss.item(), iter)
            tb_logger.add_scalar('Loss/color_loss', loss_output['rgb_loss'].item(), iter)
            tb_logger.add_scalar('Loss/eikonal_loss', loss_output['eikonal_loss'].item(), iter)
            tb_logger.add_scalar('Loss/smooth_loss', loss_output['smooth_loss'].item(), iter)
            tb_logger.add_scalar('Loss/depth_loss', loss_output['depth_loss'].item(), iter)
            tb_logger.add_scalar('Loss/normal_l1_loss', loss_output['normal_l1'].item(), iter)
            tb_logger.add_scalar('Loss/normal_cos_loss', loss_output['normal_cos'].item(), iter)
            tb_logger.add_scalar('Loss/ray_mask_loss', loss_output['ray_mask_loss'].item(), iter)
            tb_logger.add_scalar('Loss/instance_mask_loss', loss_output['instance_mask_loss'].item(), iter)
            tb_logger.add_scalar('Loss/sdf_loss', loss_output['sdf_loss'].item(), iter)
            tb_logger.add_scalar('Loss/vis_sdf_loss', loss_output['vis_sdf_loss'].item(), iter)
            tb_logger.add_scalar('Loss/grad_norm', total_norm, iter)

            tb_logger.add_scalar('Statistics/beta', model.module.density.get_beta().item(), iter)
            tb_logger.add_scalar('Statistics/alpha', 1. / model.module.density.get_beta().item(), iter)
            tb_logger.add_scalar('Statistics/psnr', psnr.item(), iter)
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            tb_logger.add_scalar("train/lr", current_lr, iter)

            iter += 1
        
        # 调整学习率
        avg_total_prob = total_prob /  (batch_id + 1)


        scheduler.step()

        # after model_save_interval epoch, evaluate the model
        if e % config['other']['model_save_interval'] == 0:
            model.eval()
            eval_loss = 0
            tea_eval_loss = 0

            eval_loss_info = {
            }

            tea_eval_loss_info = {
            }
            cfg.log_string("Switch Phase to Test")
            for batch_id, (indices, model_input, ground_truth) in enumerate(test_loader):
                torch.cuda.empty_cache()
            # 确保所有数据在 GPU 上
                model_input = {k: v.cuda().to(torch.float32) for k, v in model_input.items()}
                ground_truth = {k: v.cuda().to(torch.float32) if isinstance(v, torch.Tensor) else v for k, v in ground_truth.items()}
                model_outputs = model(model_input, indices)
                tea_model_outputs = tea_net(model_input, 
                                            indices)

                loss_output = loss(model_outputs, ground_truth, e)
                tea_loss_output = loss(tea_model_outputs, ground_truth, e)

                total_loss = loss_output['total_loss']
                tea_total_loss = tea_loss_output['total_loss']
            

                psnr = get_psnr(model_outputs['rgb_values'], ground_truth['rgb'].cuda().reshape(-1,3))
                tea_psnr = get_psnr(tea_model_outputs['rgb_values'], ground_truth['rgb'].cuda().reshape(-1,3))

                msg = 'Validation_Student {:0>8},[epoch {}] ({}/{}): total_loss = {}, rgb_loss = {}, eikonal_loss = {}, depth_loss = {}, normal_l1 = {}, normal_cos = {}, ray_mask_loss = {}, instance_mask_loss = {}, sdf_loss = {}, vis_sdf_loss = {}, psnr = {}, bete={}, alpha={}'.format(
                    str(datetime.timedelta(seconds=round(time.time() - start_t))),
                    e,
                    batch_id + 1,
                    len(test_loader),
                    total_loss.item(),
                    loss_output['rgb_loss'].item(),
                    loss_output['eikonal_loss'].item(),
                    loss_output['depth_loss'].item(),
                    loss_output['normal_l1'].item(),
                    loss_output['normal_cos'].item(),
                    loss_output['ray_mask_loss'].item(),
                    loss_output['instance_mask_loss'].item(),
                    loss_output['sdf_loss'].item(),
                    loss_output['vis_sdf_loss'].item(),
                    psnr.item(),
                    model.module.density.get_beta().item(),
                    1. / model.module.density.get_beta().item()
                )
                cfg.log_string(msg)

                msg = 'Validation_Teacher {:0>8},[epoch {}] ({}/{}): total_loss = {}, rgb_loss = {}, eikonal_loss = {}, depth_loss = {}, normal_l1 = {}, normal_cos = {}, ray_mask_loss = {}, instance_mask_loss = {}, sdf_loss = {}, vis_sdf_loss = {}, psnr = {}, bete={}, alpha={}'.format(
                    str(datetime.timedelta(seconds=round(time.time() - start_t))),
                    e,
                    batch_id + 1,
                    len(test_loader),
                    tea_total_loss.item(),
                    tea_loss_output['rgb_loss'].item(),
                    tea_loss_output['eikonal_loss'].item(),
                    tea_loss_output['depth_loss'].item(),
                    tea_loss_output['normal_l1'].item(),
                    tea_loss_output['normal_cos'].item(),
                    tea_loss_output['ray_mask_loss'].item(),
                    tea_loss_output['instance_mask_loss'].item(),
                    tea_loss_output['sdf_loss'].item(),
                    tea_loss_output['vis_sdf_loss'].item(),
                    tea_psnr.item(),
                    tea_net.density.get_beta().item(),
                    1. / tea_net.density.get_beta().item()
                )
                cfg.log_string(msg)

                for key in loss_output:
                    if "total" not in key:
                        if key not in eval_loss_info:
                            eval_loss_info[key] = 0
                        eval_loss_info[key] += torch.mean(loss_output[key]).item()

                for key in tea_loss_output:
                    if "total" not in key:
                        if key not in tea_eval_loss_info:
                            tea_eval_loss_info[key] = 0
                        tea_eval_loss_info[key] += torch.mean(tea_loss_output[key]).item()


                if torch.isnan(total_loss).item():
                    print("NaN detected in total_loss!")
                else:
                    eval_loss += total_loss.item()

                if torch.isnan(tea_total_loss).item():
                    print("NaN detected in tea_total_loss!")
                else:
                    tea_eval_loss += tea_total_loss.item()

            # 输出该epoch训练的教师模型数据情况
            total_prob_msg = f'Trainning message: the key value for prob is {filt_val} , and avg_total_prob is {avg_total_prob} ,  the number of total unlabel data which is successfully trained is {total_unlabel}' 
            cfg.log_string(total_prob_msg)     

            # 计算loss 用于决定是否保存模型                
            avg_eval_loss = eval_loss / (batch_id + 1)
            tea_avg_eval_loss = tea_eval_loss / (batch_id + 1)

            for key in eval_loss_info:
                eval_loss_info[key] = eval_loss_info[key] / (batch_id + 1)
            eval_loss_msg = f'avg_eval_loss is {avg_eval_loss}'
            cfg.log_string(eval_loss_msg)
            tb_logger.add_scalar('eval/eval_loss', avg_eval_loss, e)
            for key in eval_loss_info:
                tb_logger.add_scalar("eval/" + key, eval_loss_info[key], e)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_eval_loss)


            for key in tea_eval_loss_info:
                tea_eval_loss_info[key] = tea_eval_loss_info[key] / (batch_id + 1)
            tea_eval_loss_msg = f'tea_avg_eval_loss is {tea_avg_eval_loss}'
            cfg.log_string(tea_eval_loss_msg)
            tb_logger.add_scalar('eval/tea_eval_loss', tea_avg_eval_loss, e)
            for key in tea_eval_loss_info:
                tb_logger.add_scalar("eval/tea_" + key, tea_eval_loss_info[key], e)
            # if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            #     scheduler.step(tea_avg_eval_loss)            

                
            # else:
            #     scheduler.step()


            # 按照 best 和latest 来保存模型
            # 保存学生模型
            # checkpoint.register_modules(epoch=e, iter=iter, min_loss=avg_eval_loss)
            # if avg_eval_loss < min_eval_loss:
            #     checkpoint.save('best')
            #     min_eval_loss = avg_eval_loss
            # else:
            #     checkpoint.save("latest")

            # # 保存教师模型
            # if tea_avg_eval_loss < min_eval_loss_teacher:
            #     checkpoint.save('tea_model_best', tea_net)
            #     min_eval_loss_teacher = tea_avg_eval_loss
            # else:
            #     checkpoint.save("tea_model_latest", tea_net)


            # 按照 epoch 数量动态创建文件夹来保存模型
            # 动态创建文件夹路径
            epoch_folder = os.path.join(log_dir, f'train_epoch_{e}')
            best_folder = os.path.join(log_dir, 'train_best')

            # 确保文件夹存在
            os.makedirs(epoch_folder, exist_ok=True)
            os.makedirs(best_folder, exist_ok=True)

            # 保存当前 epoch 的学生模型和教师模型
            stu_model_path = os.path.join(epoch_folder, 'stu_model.pth')
            tea_model_path = os.path.join(epoch_folder, 'tea_model.pth')
            torch.save(model.state_dict(), stu_model_path)
            torch.save(tea_net.state_dict(), tea_model_path)
            cfg.log_string(f"Student model saved to {stu_model_path}")
            cfg.log_string(f"Teacher model saved to {tea_model_path}")

            # 保存最佳学生模型
            if avg_eval_loss < min_eval_loss:
                best_stu_model_path = os.path.join(best_folder, 'stu_model_best.pth')
                torch.save(model.state_dict(), best_stu_model_path)
                min_eval_loss = avg_eval_loss
                cfg.log_string(f"Best student model updated: {best_stu_model_path}")

            # 保存最佳教师模型
            if tea_avg_eval_loss < min_eval_loss_teacher:
                best_tea_model_path = os.path.join(best_folder, 'tea_model_best.pth')
                torch.save(tea_net.state_dict(), best_tea_model_path)
                min_eval_loss_teacher = tea_avg_eval_loss
                cfg.log_string(f"Best teacher model updated: {best_tea_model_path}")


            # 保存判别器模型
            discriminator_checkpoint_path = os.path.join(log_dir, 'dict_latest.pth')
            torch.save(discriminator.state_dict(), discriminator_checkpoint_path)
            cfg.log_string(f"Discriminator model saved to {discriminator_checkpoint_path}")  

            # 教师模型的 EMA 更新 在每个 batch 的末尾更新教师模型权重
            ema_update(tea_net, model, step=e, total_steps=config['other']['nepoch'])          

                    
            

