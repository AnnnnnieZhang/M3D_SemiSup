import os
import time
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils.rend_util import get_psnr
from SSP3D.discriminator import Discriminator

from utils.sdf_util import get_sdf_gt_worldcoords


def resize_sdf(batch_size,num_points_per_dim,real_sdf,fake_sdf):
    # 假设 real_sdf 和 fake_sdf 是 (batch_size * num_points, 1)

    num_points = real_sdf.shape[0]//batch_size

    real_sdf = real_sdf.view(batch_size, num_points, 1)
    fake_sdf = fake_sdf.view(batch_size, num_points, 1)

    # 假设点的 3D 坐标为以下形式 (batch_size, num_points, 3)
    coords = torch.randint(0, num_points_per_dim, (batch_size, num_points, 3)).to(real_sdf.device)  # 示例随机坐标

    # 创建规则网格
    real_sdf_grid = torch.zeros((batch_size, 1, num_points_per_dim, num_points_per_dim, num_points_per_dim)).to(real_sdf.device)
    fake_sdf_grid = torch.zeros((batch_size, 1, num_points_per_dim, num_points_per_dim, num_points_per_dim)).to(fake_sdf.device)

    # 将点采样数据填充到规则网格
    for b in range(batch_size):
        real_sdf_grid[b, 0, coords[b, :, 0], coords[b, :, 1], coords[b, :, 2]] = real_sdf[b, :, 0]
        fake_sdf_grid[b, 0, coords[b, :, 0], coords[b, :, 1], coords[b, :, 2]] = fake_sdf[b, :, 0]

    return real_sdf_grid, fake_sdf_grid


def Recon_trainer(cfg,model,loss,optimizer,scheduler,train_loader,test_loader,device,checkpoint):
    start_t = time.time()
    config = cfg.config

    log_dir = cfg.save_path
    os.makedirs(log_dir, exist_ok=True)

    cfg.write_config()
    tb_logger = SummaryWriter(log_dir)

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

    # 初始化判别器及优化器
    discriminator = Discriminator().to(device)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config['pseudo_label']['discriminator_lr'])
    lambda_adv = config['pseudo_label']['lambda_adv']
    min_disc_loss = float('inf')  # 判别器最小损失

    # 检查是否需要加载预训练判别器权重
    if config["dict_resume"] == True:
        # 拼接判别器权重文件路径
        discriminator_checkpoint_path = os.path.join(
            config["save_root_path"],
            config["exp_name"],
            config["dict_weight"]
        )

        # 尝试加载判别器权重
        if os.path.exists(discriminator_checkpoint_path):
            discriminator.load_state_dict(torch.load(discriminator_checkpoint_path, map_location=device))
            cfg.log_string(f"Discriminator model loaded from {discriminator_checkpoint_path}")
        else:
            raise FileNotFoundError(f"Discriminator checkpoint not found at {discriminator_checkpoint_path}")
    else:
        cfg.log_string("Starting training with a fresh discriminator.")


    model.train()
    discriminator.train()

    min_eval_loss = 10000
    accumulation_steps = config["data"]["accumulation_steps"]  # 累积步数，尝试不同的值

    batch_size = config['data']['batch_size']['train']  # 假设一个 batch 包含 32 个样本
    num_points_per_dim = 128  # 目标 3D 网格的边长

    
    for e in range(start_epoch, config['other']['nepoch']):
        torch.cuda.empty_cache()
        cfg.log_string("Switch Phase to Train")
        model.train()
        discriminator.train()
        optimizer.zero_grad()   # 重置优化器梯度
        for batch_id, (indices, model_input, ground_truth) in enumerate(train_loader):
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
            model_input = {k: v.cuda().to(torch.float32) for k, v in model_input.items()}
            ground_truth = {k: v.cuda().to(torch.float32) if isinstance(v, torch.Tensor) else v for k, v in ground_truth.items()}



            model_outputs = model(model_input, indices)

            # === 判别器训练 ===
            real_sdf = get_sdf_gt_worldcoords(model_outputs['sample_points'], ground_truth)  # 真实 SDF
            fake_sdf = model_outputs['sdf'].reshape(-1, 1)

            # 判别器输出

            real_sdf_grid, fake_sdf_grid = resize_sdf(batch_size,num_points_per_dim,real_sdf,fake_sdf)

            real_output = discriminator(real_sdf_grid)
            fake_output = discriminator(fake_sdf_grid.detach())

            # 判别器损失
            real_loss = torch.nn.BCELoss()(real_output, torch.ones_like(real_output))
            fake_loss = torch.nn.BCELoss()(fake_output, torch.zeros_like(fake_output))
            discriminator_loss = real_loss + fake_loss

            loss_output = loss(model_outputs, ground_truth, e)
            total_loss = loss_output['total_loss']

            # 加上对抗损失
            total_loss = total_loss + lambda_adv * fake_loss.detach()

            # 梯度累积
            # 缩放后的损失值用于反向传播和梯度累积
            scaled_total_loss = total_loss / accumulation_steps
            scaled_total_loss.backward()    
            # total_loss.backward()

            scaled_discriminator_loss = discriminator_loss / accumulation_steps
            scaled_discriminator_loss.backward()

            total_norm = 0
            '''gradient clip'''
            if (batch_id + 1) % accumulation_steps == 0:
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


                # 优化判别器
                disc_optimizer.step() 
                disc_optimizer.zero_grad()
                           

            psnr = get_psnr(model_outputs['rgb_values'], ground_truth['rgb'].cuda().reshape(-1,3))
            msg = '{:0>8},[epoch {}] ({}/{}): total_loss = {}, rgb_loss = {}, eikonal_loss = {}, depth_loss = {}, normal_l1 = {}, normal_cos = {}, ray_mask_loss = {}, instance_mask_loss = {}, sdf_loss = {}, vis_sdf_loss = {}, psnr = {}, bete={}, alpha={}, D_Loss={} '.format(
                    str(datetime.timedelta(seconds=round(time.time() - start_t))),
                    e,
                    batch_id + 1,
                    len(train_loader),
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
                    discriminator_loss.item()
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

            # tb_logger.add_scalar('Statistics/dino_weight', model_outputs['dino_weight'].item(), iter)
            tb_logger.add_scalar('Statistics/beta', model.module.density.get_beta().item(), iter)
            tb_logger.add_scalar('Statistics/alpha', 1. / model.module.density.get_beta().item(), iter)
            tb_logger.add_scalar('Statistics/psnr', psnr.item(), iter)
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            tb_logger.add_scalar("train/lr", current_lr, iter)

            iter += 1
        
        # 调整学习率
        scheduler.step()

        # after model_save_interval epoch, evaluate the model
        if e % config['other']['model_save_interval'] == 0:
            model.eval()
            discriminator.eval()
            eval_loss = 0
            total_disc_loss = 0
            eval_loss_info = {
            }
            cfg.log_string("Switch Phase to Test")
            
            for batch_id, (indices, model_input, ground_truth) in enumerate(test_loader):
                torch.cuda.empty_cache()
                model_input = {k: v.cuda().to(torch.float32) for k, v in model_input.items()}
                ground_truth = {k: v.cuda().to(torch.float32) if isinstance(v, torch.Tensor) else v for k, v in ground_truth.items()}

                model_outputs = model(model_input, indices)

                if ground_truth['rgb'].shape[0] < batch_size:
                    print(f"Skipping last batch: insufficient data ({ground_truth['rgb'].shape[0]} < {batch_size})")
                    continue

                # === 判别器训练 ===
                real_sdf = get_sdf_gt_worldcoords(model_outputs['sample_points'], ground_truth)  # 真实 SDF
                fake_sdf = model_outputs['sdf'].reshape(-1, 1)

                # 判别器输出

                real_sdf_grid, fake_sdf_grid = resize_sdf(batch_size,num_points_per_dim,real_sdf,fake_sdf)

                real_output = discriminator(real_sdf_grid)
                fake_output = discriminator(fake_sdf_grid.detach())

                # 判别器损失
                real_loss = torch.nn.BCELoss()(real_output, torch.ones_like(real_output))
                fake_loss = torch.nn.BCELoss()(fake_output, torch.zeros_like(fake_output))
                disc_loss = real_loss + fake_loss
                total_disc_loss += disc_loss.item()

                loss_output = loss(model_outputs, ground_truth, e)
                total_loss = loss_output['total_loss']

                # total_loss = total_loss + lambda_adv * fake_loss.detach()
            

                psnr = get_psnr(model_outputs['rgb_values'], ground_truth['rgb'].cuda().reshape(-1,3))
                msg = 'Validation {:0>8},[epoch {}] ({}/{}): total_loss = {}, rgb_loss = {}, eikonal_loss = {}, depth_loss = {}, normal_l1 = {}, normal_cos = {}, ray_mask_loss = {}, instance_mask_loss = {}, sdf_loss = {}, vis_sdf_loss = {}, psnr = {}, bete={}, alpha={}, D_Loss={} '.format(
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
                    1. / model.module.density.get_beta().item(),
                    disc_loss.item()
                )
                cfg.log_string(msg)

                for key in loss_output:
                    if "total" not in key:
                        if key not in eval_loss_info:
                            eval_loss_info[key] = 0
                        eval_loss_info[key] += torch.mean(loss_output[key]).item()

                if torch.isnan(total_loss).item():
                    print("NaN detected in total_loss!")
                else:
                    eval_loss += total_loss.item()
            # 计算平均  loss
            avg_disc_loss = total_disc_loss / (batch_id + 1)
            avg_eval_loss = eval_loss / (batch_id + 1)
            for key in eval_loss_info:
                eval_loss_info[key] = eval_loss_info[key] / (batch_id + 1)
            eval_loss_msg = f'avg_eval_loss is {avg_eval_loss}'
            dict_loss_msg = f'avg_dict_loss is {avg_disc_loss}'

            cfg.log_string(eval_loss_msg)
            cfg.log_string(dict_loss_msg)

            tb_logger.add_scalar('eval/eval_loss', avg_eval_loss, e)
            tb_logger.add_scalar('eval/dict_loss', avg_disc_loss, e)

            for key in eval_loss_info:
                tb_logger.add_scalar("eval/" + key, eval_loss_info[key], e)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_eval_loss)
            # else:
            #     scheduler.step()

            checkpoint.register_modules(epoch=e, iter=iter, min_loss=avg_eval_loss)
            if avg_eval_loss < min_eval_loss:
                checkpoint.save('best')
                min_eval_loss = avg_eval_loss
            else:
                checkpoint.save("latest")

            # 判别器模型保存逻辑
            if avg_disc_loss < min_disc_loss:
                min_disc_loss = avg_disc_loss
                torch.save(discriminator.state_dict(), os.path.join(log_dir, 'disc_best.pth'))
            else:
                torch.save(discriminator.state_dict(), os.path.join(log_dir, 'disc_latest.pth'))

