import os
import argparse

import cv2
import torch
import torch.nn as nn

from configs.config_utils import CONFIG
from ssr.ssr_utils.utils import load_device, get_model, get_loss, \
    get_dataloader,CheckpointIO,get_trainer,get_optimizer,load_scheduler
from ssr.ssr_utils.network_utils import fix_random_seed

from collections import OrderedDict

# for vscode debug error
dirname = os.path.dirname(cv2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path


# 去掉 'module.' 前缀
def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):  # 检查是否有 'module.' 前缀
            new_key = k[len("module."):]  # 去掉 'module.' 前缀
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('ssr training')
    parser.add_argument('--config', type=str, required=True, help='configure file for training or testing.')
    return parser.parse_args()

def run(cfg):
    torch.set_default_dtype(torch.float32)
    '''Fix random seed'''
    if cfg.config['fix_random_seed']:
        fix_random_seed(seed=1029)

    '''Load save path'''
    cfg.log_string('Data save path: %s' % (cfg.save_path))
    checkpoint=CheckpointIO(cfg)

    '''Load device'''
    cfg.log_string('Loading device settings.')
    device = load_device(cfg)

    '''Load data'''
    cfg.log_string('Loading dataset.')
    train_loader = get_dataloader(cfg.config, mode='train')
    unlabel_train_loader = get_dataloader(cfg.config, mode='train',unlabel=True)
    test_loader = get_dataloader(cfg.config, mode='val')

    '''Set loss'''
    cfg.log_string('Setting loss function')
    loss = get_loss(cfg.config, mode='train')

    '''Load net'''
    cfg.log_string('Loading model.')
    net = get_model(cfg.config, device=device).float()
    tea_net = get_model(cfg.config, device=device).float()
    checkpoint.register_modules(net=net)

    '''Load teacher_model'''
    cfg.log_string('Loading teacher model.')
    teacher_checkpoint = torch.load('pretrain_models/teachers/model_latest_5.pth')
    teacher_weights = teacher_checkpoint['net']
    teacher_weights = remove_module_prefix(teacher_checkpoint['net'])  # 处理权重
    tea_net.load_state_dict(teacher_weights)  # 加载处理后的权重
    tea_net.eval()  # 教师模型不需要训练    

    '''Load optimizer'''
    cfg.log_string('Loading optimizer.')
    optimizer = get_optimizer(config=cfg.config, net=net)
    # model, optimizer = amp.initialize(net, optimizer, opt_level="O1")
    checkpoint.register_modules(opt=optimizer)
    net = nn.DataParallel(net)
    net = net.to(device)
    checkpoint.register_modules(net=net)

    '''Load scheduler'''
    cfg.log_string('Loading optimizer scheduler.')
    scheduler = load_scheduler(config=cfg.config, optimizer=optimizer, train_loader=train_loader)
    checkpoint.register_modules(sch=scheduler)

    '''Load trainer'''
    cfg.log_string('Loading trainer.')
    trainer = get_trainer(cfg.config)

    '''Start to train'''
    cfg.log_string('Start to train.')
    if cfg.config['Stage'] == 1:
        trainer(cfg, net, loss, optimizer,scheduler,train_loader=train_loader, test_loader=test_loader,device=device,checkpoint=checkpoint)
    elif cfg.config['Stage'] == 2:
        trainer(cfg, net, loss, optimizer,scheduler,train_loader=train_loader, unlabel_train_loader = unlabel_train_loader, test_loader=test_loader,device=device,checkpoint=checkpoint, tea_net=tea_net)
    cfg.log_string('Training finished.')

if __name__=="__main__":
    args=parse_args()
    cfg=CONFIG(args.config)
    cfg.update_config(args.__dict__)

    cfg.log_string('Loading configuration')
    cfg.log_string(cfg.config)
    cfg.write_config()
    cfg.log_string('nvidia/MambaVision-B-1K, pretrained = true, depth-->Depthanything, resnte18 34 -->mamba')

    cfg.log_string('Training begin.')
    
    run(cfg)
