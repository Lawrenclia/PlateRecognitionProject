import argparse
# import logging # <-- 已删除
import math
import os
import random
import time
from pathlib import Path
from threading import Thread
from warnings import warn

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # 导入 test.py 以便在每个 epoch 后计算 mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.face_datasets import create_dataloader  # <-- 用于加载(边界框+关键点)的数据加载器
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    print_mutation, set_logging
from utils.google_utils import attempt_download
from utils.loss import compute_loss  # <-- 自定义的损失函数，包含关键点(landmark)损失
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first

begin_save=1 # 从第几个epoch开始保存模型

# 尝试导入 wandb (Weights & Biases) 用于实验跟踪
try:
    import wandb
except ImportError:
    wandb = None


def train(hyp, opt, device, tb_writer=None, wandb=None):
    # logger.info(f'Hyperparameters {hyp}') # <-- 已删除
    
    # 从 opt 和 hyp 中提取常用变量
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    # --- 目录设置 ---
    wdir = save_dir / 'weights'  # 权重保存目录
    wdir.mkdir(parents=True, exist_ok=True)  # 创建目录
    last = wdir / 'last.pt'  # 最后一次的权重
    best = wdir / 'best.pt'  # 最佳性能的权重
    results_file = save_dir / 'results.txt'  # 训练结果日志

    # --- 保存运行配置 ---
    # 将超参数 hyp 保存到 hyp.yaml
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    # 将命令行选项 opt 保存到 opt.yaml
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # --- 基本配置 ---
    plots = not opt.evolve  # 是否创建图表 (在 "进化" 超参数时关闭)
    cuda = device.type != 'cpu'  # 判断是否使用 CUDA
    init_seeds(2 + rank)  # 初始化随机种子
    
    # 加载数据配置文件 (data.yaml)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    
    # 检查数据集 (仅在 rank 0 进程中执行)
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)
        
    train_path = data_dict['train']  # 训练集路径
    test_path = data_dict['val']    # 验证集路径
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # 类别数量
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # 类别名称
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # 检查类别名和数量是否匹配

    # --- 模型 ---
    pretrained = weights.endswith('.pt')  # 判断是否使用预训练权重
    if pretrained:
        # 加载预训练模型
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # 如果本地没有，尝试下载
        ckpt = torch.load(weights, map_location=device)  # 加载 checkpoint
        if hyp.get('anchors'):
            ckpt['model'].yaml['anchors'] = round(hyp['anchors'])  # 强制使用新的 anchors
        
        # 创建模型
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc).to(device)
        exclude = ['anchor'] if opt.cfg or hyp.get('anchors') else []  # 要排除的权重
        state_dict = ckpt['model'].float().state_dict()  # 预训练模型的权重
        
        # 智能加载权重：只加载层名和大小都匹配的权重
        # 这对于迁移学习至关重要
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)
        model.load_state_dict(state_dict, strict=False)  # 加载
        # logger.info('Transferred %g/%g items from %s' % ...) # <-- 已删除
    else:
        # 从头创建新模型
        model = Model(opt.cfg, ch=3, nc=nc).to(device)

    # --- 冻结层 (Freeze) ---
    # freeze = []  # 在这里指定要冻结的层
    # for k, v in model.named_parameters():
    #     v.requires_grad = True  # 默认训练所有层
    #     if any(x in k for x in freeze):
    #         print('freezing %s' % k)
    #         v.requires_grad = False

    # --- 优化器 (Optimizer) ---
    nbs = 64  # 基准 batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # 梯度累积步数
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # 根据 batch size 缩放权重衰减

    pg0, pg1, pg2 = [], [], []  # 定义三个参数组
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  #  biases (偏置)
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # BatchNorm 权重 (不使用 weight_decay)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # 其他权重 (使用 weight_decay)

    # 根据 opt.adam 选择 Adam 或 SGD 优化器
    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # 添加 pg1 (带衰减)
    optimizer.add_param_group({'params': pg2})  # 添加 pg2 (不带衰减)
    # logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % ...) # <-- 已删除
    del pg0, pg1, pg2

    # --- 学习率调度器 (Scheduler) ---
    # 使用余弦退火策略
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp['lrf']) + hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # --- 日志记录器 (Logging) ---
    if wandb and wandb.run is None:
        opt.hyp = hyp  # 添加超参数到 wandb 配置
        wandb_run = wandb.init(config=opt, resume="allow",
                               project='YOLOv5' if opt.project == 'runs/train' else Path(opt.project).stem,
                               name=save_dir.stem,
                               id=ckpt.get('wandb_id') if 'ckpt' in locals() else None)
    loggers = {'wandb': wandb}  # 日志记录器字典

    # --- 恢复训练 (Resume) ---
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # 恢复优化器状态
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = 0 # ckpt['best_fitness']

        # 恢复训练结果
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])

        # 恢复 epoch
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s a ' % (weights)
        if epochs < start_epoch:
            # logger.info('%s has been trained for %g epochs...' % ...) # <-- 已删除
            epochs += ckpt['epoch']  # 增加 finetune 的 epochs

        del ckpt, state_dict

    # --- 图像尺寸设置 ---
    gs = int(max(model.stride))  # 获取模型最大 stride
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # 确保图像尺寸是 stride 的倍数

    # --- DDP / DataParallel 设置 ---
    # DP (DataParallel, 单机多卡)
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm (用于 DDP)
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        # logger.info('Using SyncBatchNorm()') # <-- 已删除

    # EMA (Exponential Moving Average)
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # DDP (DistributedDataParallel, 多机多卡)
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)

    # --- 数据加载器 (Dataloader) ---
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights)
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # 最大的类别索引
    nb = len(dataloader)  # batch 的数量
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s...' % (mlc, nc, opt.data, nc - 1)

    # --- 主进程 (Rank 0) 设置 ---
    if rank in [-1, 0]:
        ema.updates = start_epoch * nb // accumulate  # 恢复 EMA 更新次数
        
        # 创建验证集加载器
        testloader = create_dataloader(test_path, imgsz_test, total_batch_size, gs, opt,
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True,
                                       rank=-1, world_size=opt.world_size, workers=opt.workers, pad=0.5)[0]

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            if plots:
                plot_labels(labels, save_dir, loggers)  # 绘制标签分布图
                if tb_writer:
                    tb_writer.add_histogram('classes', torch.tensor(labels[:, 0]), 0)

            # 自动计算 Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

    # --- 模型参数附加 ---
    hyp['cls'] *= nc / 80.  # 根据类别数缩放 'cls' 超参数
    model.nc = nc  # 将类别数附加到模型
    model.hyp = hyp  # 将超参数附加到模型
    model.gr = 1.0  # iou loss ratio
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # 附加类别权重
    model.names = names  # 附加类别名称

    # --- 开始训练 ---
    t0 = time.time()
    # 预热迭代次数 (warmup iterations)
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # 设置调度器起始 epoch
    scaler = amp.GradScaler(enabled=cuda)  # 启用自动混合精度 (AMP)
    
    # logger.info('Image sizes %g train, %g test\n' ... ) # <-- 已删除

    # --- 训练循环 (Epoch Loop) ---
    for epoch in range(start_epoch, epochs):
        model.train()  # 设置模型为训练模式

        # --- 图像权重 (Image Weights) ---
        if opt.image_weights:
            # (可选) 根据 mAP 动态调整样本权重
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)
            # DDP 广播
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # mloss 数组，包含5个损失: box, obj, cls, landmark, total
        # 这是针对车牌(关键点)检测的定制
        mloss = torch.zeros(5, device=device)  # mean losses
        
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
            
        pbar = enumerate(dataloader)
        # 打印训练表头 (已从 logger 改为 print)
        if rank in [-1, 0]:
            print(('\n' + '%10s' * 9) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'landmark', 'total', 'targets', 'img_size'))
            pbar = tqdm(pbar, total=nb)  # 封装为 tqdm 进度条
            
        optimizer.zero_grad()
        
        # --- 批次循环 (Batch Loop) ---
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch  # global number of iterations
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # 图像归一化

            # --- 预热 (Warmup) ---
            if ni <= nw:
                xi = [0, nw]  # x interp
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr 从 0.1 降到 lr0, other lrs 从 0.0 升到 lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # --- 多尺度训练 (Multi-scale) ---
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # 随机尺寸
                sf = sz / max(imgs.shape[2:])  # 缩放因子
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # --- 前向传播 (Forward) ---
            with amp.autocast(enabled=cuda):  # 自动混合精度
                pred = model(imgs)  # 模型推理
                
                # 计算损失 (包括了关键点损失)
                loss, loss_items = compute_loss(pred, targets.to(device), model)
                if rank != -1:
                    loss *= opt.world_size  # DDP 模式下缩放损失

            # --- 反向传播 (Backward) ---
            scaler.scale(loss).backward()

            # --- 优化器步进 (Optimize) ---
            if ni % accumulate == 0:
                scaler.step(optimizer)  # 更新优化器
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model) # 更新 EMA 模型

            # --- 打印信息 (Print) ---
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # 更新平均损失
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # 显存
                
                # 构造进度条显示字符串
                # 这里的 *mloss 会自动解包 5 个损失值
                s = ('%10s' * 2 + '%10.4g' * 7) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)  # 更新 tqdm 进度条

                # --- 绘图 (Plot) ---
                # 保存前3个 batch 的训练图像
                if plots and ni < 3:
                    f = save_dir / f'train_batch{ni}.jpg'
                    Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                elif plots and ni == 3 and wandb:
                    wandb.log({"Mosaics": [wandb.Image(str(x), caption=x.name) for x in save_dir.glob('train*.jpg')]})

            # --- 结束批次循环 ---
        # --- 结束 Epoch 循环 ---

        # --- 学习率调度器步进 ---
        lr = [x['lr'] for x in optimizer.param_groups]  # 记录当前学习率
        scheduler.step()

        # --- 验证与保存 (Validation & Save) ---
        if rank in [-1, 0] and epoch >= begin_save: # 从 begin_save 开始保存和验证
            # mAP
            if ema:
                ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            
            # 运行验证 (如果 notest=False 或者这是最后一轮)
            if not opt.notest or final_epoch:
                results, maps, times = test.test(opt.data,
                                                 batch_size=total_batch_size,
                                                 imgsz=imgsz_test,
                                                 model=ema.ema,  # 使用 EMA 模型进行验证
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=save_dir,
                                                 plots=False,
                                                 log_imgs=opt.log_imgs if wandb else 0)

            # --- 写入结果 ---
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP@.5, mAP@.5-.95, ...
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # --- 日志 (Log) ---
            # 注意: 'train/landmark_loss' 是车牌关键点定制版
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss', 'train/landmark_loss',
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',
                    'x/lr0', 'x/lr1', 'x/lr2']
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # 写入 TensorBoard
                if wandb:
                    wandb.log({tag: x})  # 写入 W&B

            # --- 更新最佳 mAP ---
            fi = fitness(np.array(results).reshape(1, -1))  # 计算适应度 (mAP 的加权)
            if fi > best_fitness:
                best_fitness = fi

            # --- 保存模型 ---
            save = (not opt.nosave) or (final_epoch and not opt.evolve)
            if save:
                with open(results_file, 'r') as f:  # 创建 checkpoint
                    ckpt = {'epoch': epoch,
                            'best_fitness': best_fitness,
                            'training_results': f.read(),
                            'model': ema.ema,  # 保存 EMA 模型
                            'optimizer': None if final_epoch else optimizer.state_dict(),
                            'wandb_id': wandb_run.id if wandb else None}

                # 保存 last.pt
                torch.save(ckpt, last)
                # 如果是最佳模型，保存 best.pt
                if best_fitness == fi:
                    ckpt_best = {
                            'epoch': epoch,
                            'best_fitness': best_fitness,
                            'model': ema.ema,
                            }
                    torch.save(ckpt_best, best)
                del ckpt
        # --- 结束 Epoch 循环 ---
    # --- 结束训练 ---

    # --- 训练收尾 ---
    if rank in [-1, 0]:
        # 剥离优化器 (Strip optimizers)
        final = best if best.exists() else last
        for f in [last, best]:
            if f.exists():
                strip_optimizer(f)  # 移除优化器状态，减小模型大小
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  # 上传到 GCS

        # 绘制结果图
        if plots:
            plot_results(save_dir=save_dir)
            if wandb:
                files = ['results.png', 'precision_recall_curve.png', 'confusion_matrix.png']
                wandb.log({"Results": [wandb.Image(str(save_dir / f), caption=f) for f in files
                                       if (save_dir / f).exists()]})
                if opt.log_artifacts:
                    wandb.log_artifact(artifact_or_path=str(final), type='model', name=save_dir.stem)

        # (可选) 在 COCO 数据集上测试
        # logger.info('%g epochs completed in %.3f hours.\n' % ...) # <-- 已删除
        if opt.data.endswith('coco.yaml') and nc == 80:
            for conf, iou, save_json in ([0.25, 0.45, False], [0.001, 0.65, True]):
                results, _, _ = test.test(opt.data,
                                          batch_size=total_batch_size,
                                          imgsz=imgsz_test,
                                          conf_thres=conf,
                                          iou_thres=iou,
                                          model=attempt_load(final, device).half(),
                                          single_cls=opt.single_cls,
                                          dataloader=testloader,
                                          save_dir=save_dir,
                                          save_json=save_json,
                                          plots=False)

    else:
        dist.destroy_process_group()

    wandb.run.finish() if wandb and wandb.run else None
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # --- 关键参数 ---
    parser.add_argument('--weights', type=str, default='weights/yolov5n.pt', help='初始权重路径 (例如 yolov5n.pt)')
    parser.add_argument('--cfg', type=str, default='models/yolov5n-plates.yaml', help='模型配置文件 (必须是包含关键点的版本)')
    parser.add_argument('--data', type=str, default='data/ccpd.yaml', help='数据集配置文件 (指向你 ccpd_process.py 生成的数据)')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='超参数文件')
    
    # --- 训练控制参数 ---
    parser.add_argument('--epochs', type=int, default=120, help='训练总轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='所有 GPU 的总 batch size')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[训练, 测试] 图像尺寸')
    parser.add_argument('--rect', action='store_true', help='矩形训练 (节省显存)')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='从last.pt 恢复训练')
    parser.add_argument('--nosave', action='store_true', help='只保存最终的 checkpoint')
    parser.add_argument('--notest', action='store_true', help='只在最后一轮测试')
    parser.add_argument('--noautoanchor', action='store_true', help='禁用自动 anchor 检查')
    parser.add_argument('--evolve', action='store_true', help='(高级) 进化超参数')
    parser.add_argument('--cache-images', action='store_true', help='缓存图像到内存以加速训练')
    parser.add_argument('--image-weights', action='store_true', help='(高级) 使用加权图像采样')
    
    # --- 硬件与分布式参数 ---
    parser.add_argument('--device', default='', help='cuda 设备, e.g. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', default=True, help='多尺度训练 (+/- 50%%)')
    parser.add_argument('--single-cls', action='store_true', help='把多类数据当作单类训练')
    parser.add_argument('--adam', action='store_true', help='使用 Adam 优化器')
    parser.add_argument('--sync-bn', action='store_true', help='(DDP) 使用 SyncBatchNorm')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP 参数, 请勿修改')
    parser.add_argument('--workers', type=int, default=4, help='数据加载器 worker 数量')

    # --- 实验与日志参数 ---
    parser.add_argument('--project', default='runs/train', help='保存到 project/name 目录')
    parser.add_argument('--name', default='plate_exp', help='保存到 project/name 目录 (实验名称)')
    parser.add_argument('--exist-ok', action='store_true', help='允许覆盖已存在的实验目录')
    parser.add_argument('--log-imgs', type=int, default=16, help='W&B 日志图像数量')
    parser.add_argument('--log-artifacts', action='store_true', help='W&B 日志模型文件')
    parser.add_argument('--bucket', type=str, default='', help='(GCS) gsutil bucket')
    opt = parser.parse_args()

    # --- DDP (分布式) 环境变量设置 ---
    opt.total_batch_size = opt.batch_size
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    # set_logging(opt.global_rank) # <-- 已删除

    # --- 恢复训练 (Resume) 逻辑 ---
    if opt.resume:  # 恢复被中断的训练
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # 找到最新的 checkpoint
        assert os.path.isfile(ckpt), '错误: --resume checkpoint 不存在'
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))  # 替换 opt
        opt.cfg, opt.weights, opt.resume = '', ckpt, True
        # logger.info('Resuming training from %s' % ckpt) # <-- 已删除
    else:
        # 检查配置文件
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)
        assert len(opt.cfg) or len(opt.weights), '必须指定 --cfg 或 --weights'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # 扩展 img_size
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # 递增实验目录

    # --- DDP (分布式) 后端设置 ---
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # 初始化 DDP
        assert opt.batch_size % opt.world_size == 0, '--batch-size 必须是 GPU 数量的倍数'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # --- 加载超参数 ---
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    # --- 开始训练 ---
    # logger.info(opt) # <-- 已删除
    if not opt.evolve:
        tb_writer = None  # 初始化 TensorBoard
        if opt.global_rank in [-1, 0]:
            # logger.info(f'Start Tensorboard with "tensorboard --logdir {opt.project}"') # <-- 已删除
            tb_writer = SummaryWriter(opt.save_dir)  # 创建 tb_writer
        
        # *** 调用核心训练函数 ***
        train(hyp, opt, device, tb_writer, wandb)

    # --- (可选) 进化超参数 ---
    else:
        # ... (进化超参数的逻辑，这里保持不变) ...
        meta = {'lr0': (1, 1e-5, 1e-1),  # ...
                'mixup': (1, 0.0, 1.0)}

        assert opt.local_rank == -1, 'DDP 模式不支持 --evolve'
        opt.notest, opt.nosave = True, True
        yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)

        for _ in range(300):  # 进化代数
            if Path('evolve.txt').exists():
                # ... (选择父代和变异) ...
                parent = 'single'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))
                x = x[np.argsort(-fitness(x))][:n]
                w = fitness(x) - fitness(x).min()
                if parent == 'single' or len(x) == 1:
                    x = x[random.choices(range(n), weights=w)[0]]
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()

                # 变异
                mp, s = 0.8, 0.2
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):
                    hyp[k] = float(x[i + 7] * v[i])

            # 约束范围
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])
                hyp[k] = min(hyp[k], v[2])
                hyp[k] = round(hyp[k], 5)

            # 训练变异体
            results = train(hyp.copy(), opt, device, wandb=wandb)

            # 写入变异结果
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # 绘制进化结果
        plot_evolution(yaml_file)
        print(f'超参数进化完成. 最佳结果保存在: {yaml_file}\n'
              f'使用最佳超参数训练: $ python train.py --hyp {yaml_file}')
