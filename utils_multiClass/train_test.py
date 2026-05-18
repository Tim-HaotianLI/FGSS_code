import random

import torch
import numpy as np
import pandas as pd
from tqdm import trange

from utils_multiClass.optimizer import get_optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from utils_multiClass.evaluation import calculate_metrics, confusion_stats, evaluate_multitask


def unfreeze_backbone_gradually(model, epoch, total_epochs):
    """渐进式解冻Backbone: 三阶段策略"""
    backbone_name = model.object_extractor.backbone_name.lower()
    progress = epoch / total_epochs

    # 先冻结所有
    for param in model.object_extractor.parameters():
        param.requires_grad = False
    if hasattr(model, 'global_extractor'):
        for param in model.global_extractor.parameters():
            param.requires_grad = False

    if progress >= 0.3:  # 30%后开始解冻
        _unfreeze_last_stage(model.object_extractor, backbone_name)
        if hasattr(model, 'global_extractor'):
            _unfreeze_last_stage(model.global_extractor, backbone_name)

    # if progress >= 0.6:  # 60%后解冻第二阶段
    #     _unfreeze_second_last_stage(model.object_extractor, backbone_name)
    #     if hasattr(model, 'global_extractor'):
    #         _unfreeze_second_last_stage(model.global_extractor, backbone_name)

    # if progress >= 0.8:  # 80%后全部解冻（可选）
    #     for param in model.object_extractor.parameters():
    #         param.requires_grad = True
    #     if hasattr(model, 'global_extractor'):
    #         for param in model.global_extractor.parameters():
    #             param.requires_grad = True


def _unfreeze_last_stage(extractor, backbone_name):
    """解冻最后一个stage"""

    if "resnet" in backbone_name:
        for param in extractor.features[7].parameters():
            param.requires_grad = True
    elif "convnext" in backbone_name:
        for param in extractor.features[0][7].parameters():
            param.requires_grad = True
    elif "swin" in backbone_name:
        for param in extractor.features.features[7].parameters():
            param.requires_grad = True
        for param in extractor.features.norm.parameters():
            param.requires_grad = True
    elif "vit" in backbone_name:
        for param in extractor.features.encoder.layers[11].parameters():
            param.requires_grad = True
        for param in extractor.features.encoder.ln.parameters():
            param.requires_grad = True


def _unfreeze_second_last_stage(extractor, backbone_name):
    if "resnet" in backbone_name:
        for param in extractor.features[6].parameters():
            param.requires_grad = True
    elif "convnext" in backbone_name:
        for param in extractor.features[0][6].parameters():
            param.requires_grad = True
    elif "swin" in backbone_name:
        for param in extractor.features.features[6].parameters():
            param.requires_grad = True
        # for param in extractor.features.norm.parameters():
        #     param.requires_grad = True
    elif "vit" in backbone_name:
        for param in extractor.features.encoder.layers[10].parameters():
            param.requires_grad = True
        # for param in extractor.features.encoder.ln.parameters():
        #     param.requires_grad = True
    pass


def mixup_data(x1, x2, y, alpha=0.5):
    """
    Mixup数据增强

    参数:
        x1: img_global [B, C, H, W]
        x2: img_crop [B, C, H, W]
        y: labels [B]
        alpha: Beta分布参数，控制混合程度

    返回:
        mixed_x1: 混合后的global图像
        mixed_x2: 混合后的crop图像
        y_a: 第一个样本的标签
        y_b: 第二个样本的标签
        lam: 混合系数
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size(0)
    index = torch.randperm(batch_size).cuda()

    # 混合两个输入
    mixed_x1 = lam * x1 + (1 - lam) * x1[index]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index]

    y_a, y_b = y, y[index]

    return mixed_x1, mixed_x2, y_a, y_b, lam

def progressive_unfreeze_vit_with_scheduler(model, optimizer, epoch, scheduler = None):
    vit = model.object_extractor.features # vit主体
    vit_g = model.global_extractor.features # vit主体
    rebuild_scheduler = False

    if epoch == 2: # 2
        print("🧩 解冻 ViT local encoder.layers[10,11] (高层语义)")
        no_decay = []
        decay = []
        for i, blk in enumerate(vit.encoder.layers):
            # if i >= 8:
            if i in [10,11]:
                for name, p in blk.named_parameters():
                    p.requires_grad = True
                    if any(nd in name for nd in ["bias", "norm"]):
                        no_decay.append(p)
                    else:
                        decay.append(p)
        if decay and no_decay:
            lr = 5e-5
            optimizer.add_param_group({
                "params":decay,
                "lr": lr, # 2e-6
                "weight_decay": 0.05,
                "name": "high_layers_decay"  # 🔥 给参数组命名
            })

            optimizer.add_param_group({
                "params": no_decay,
                "lr": lr,  # 2e-6
                "weight_decay": 0.0,
                "name": "high_layers_no_decay"  # 🔥 给参数组命名
            })

            print(f"✅ 已新增object extractor参数 {len(decay) + len(no_decay)} 个参数到优化器。  lr: {lr}")
            rebuild_scheduler = True

        no_decay_g = []
        decay_g = []
        for i, blk in enumerate(vit_g.encoder.layers):
            # if i >= 8:
            if i in [10, 11]:
                for name, p in blk.named_parameters():
                    p.requires_grad = True
                    if any(nd in name for nd in ["bias", "norm"]):
                        no_decay_g.append(p)
                    else:
                        decay_g.append(p)
        if decay_g and no_decay_g:
            lr_g = 5e-5 / 2
            optimizer.add_param_group({
                "params":decay_g,
                "lr": lr_g, # 2e-6
                "weight_decay": 0.05,
                "name": "high_layers_global_decay"  # 🔥 给参数组命名
            })

            optimizer.add_param_group({
                "params": no_decay_g,
                "lr": lr_g,  # 2e-6
                "weight_decay": 0.0,
                "name": "high_layers_global_nodecay"  # 🔥 给参数组命名
            })
            print(f"✅ 已新增global extractor参数 {len(decay_g) + len(no_decay_g)} 个参数到优化器。  lr: {lr_g}")

        # 阶段3：再解冻中层
    elif epoch == 6: # 10
        print("🧩 解冻 ViT local encoder.layers[8,9] (中高层特征)")
        no_decay = []
        decay = []
        for i, blk in enumerate(vit.encoder.layers):
            # if 4 <= i < 8:
            if i in [8,9]:
                for name, p in blk.named_parameters():
                    p.requires_grad = True
                    if any(nd in name for nd in ["bias", "norm"]):
                        no_decay.append(p)
                    else:
                        decay.append(p)
        if decay and no_decay:
            lr = 8e-5
            optimizer.add_param_group({
                "params": decay,
                "lr": lr,  # 2e-6
                "weight_decay": 0.05,
                "name": "high_middle_layers_decay"  # 🔥 给参数组命名
            })

            optimizer.add_param_group({
                "params": no_decay,
                "lr": lr,  # 2e-6
                "weight_decay": 0.0,
                "name": "high_middle_layers_nodecay"  # 🔥 给参数组命名
            })

            print(f"✅ 已新增object extractor参数 {len(decay) + len(no_decay)} 个参数到优化器。  lr: {lr}")
            rebuild_scheduler = True

        no_decay_g = []
        decay_g = []
        for i, blk in enumerate(vit_g.encoder.layers):
            # if i >= 8:
            if i in [8,9]:
                for name, p in blk.named_parameters():
                    p.requires_grad = True
                    if any(nd in name for nd in ["bias", "norm"]):
                        no_decay_g.append(p)
                    else:
                        decay_g.append(p)
        if decay_g and no_decay_g:
            lr_g = 8e-5 / 2
            optimizer.add_param_group({
                "params": decay_g,
                "lr": lr_g,  # 2e-6
                "weight_decay": 0.05,
                "name": "high_middle_layers_global_decay"  # 🔥 给参数组命名
            })

            optimizer.add_param_group({
                "params": no_decay_g,
                "lr": lr_g,  # 2e-6
                "weight_decay": 0.0,
                "name": "high_middle_layers_global_nodecay"  # 🔥 给参数组命名
            })
            print(f"✅ 已新增global extractor参数 {len(decay_g) + len(no_decay_g)} 个参数到优化器。  lr: {lr_g}")

    elif epoch == 10: # 10
        print("🧩 解冻 ViT local encoder.layers[6,7] (中高层特征)")
        no_decay = []
        decay = []
        for i, blk in enumerate(vit.encoder.layers):
            # if 4 <= i < 8:
            if i in [6,7]:
                for name, p in blk.named_parameters():
                    p.requires_grad = True
                    if any(nd in name for nd in ["bias", "norm"]):
                        no_decay.append(p)
                    else:
                        decay.append(p)
        if decay and no_decay:
            lr = 1e-4
            optimizer.add_param_group({
                "params": decay,
                "lr": lr,  # 2e-6
                "weight_decay": 0.05,
                "name": "middle_layers_deacy"  # 🔥 给参数组命名
            })

            optimizer.add_param_group({
                "params": no_decay,
                "lr": lr,  # 2e-6
                "weight_decay": 0.0,
                "name": "middle_layers_nodeacy"  # 🔥 给参数组命名
            })

            print(f"✅ 已新增object extractor参数 {len(decay) + len(no_decay)} 个参数到优化器。  lr: {lr}")
            rebuild_scheduler = True

        no_decay_g = []
        decay_g = []
        for i, blk in enumerate(vit_g.encoder.layers):
            # if i >= 8:
            if i in [6, 7]:
                for name, p in blk.named_parameters():
                    p.requires_grad = True
                    if any(nd in name for nd in ["bias", "norm"]):
                        no_decay_g.append(p)
                    else:
                        decay_g.append(p)
        if decay_g and no_decay_g:
            lr_g = 1e-4 / 2
            optimizer.add_param_group({
                "params": decay_g,
                "lr": lr_g,  # 2e-6
                "weight_decay": 0.05,
                "name": "middle_layers_global_deacy"  # 🔥 给参数组命名
            })

            optimizer.add_param_group({
                "params": no_decay_g,
                "lr": lr_g,  # 2e-6
                "weight_decay": 0.0,
                "name": "middle_layers_global_nodecay"  # 🔥 给参数组命名
            })
            print(f"✅ 已新增global extractor参数 {len(decay_g) + len(no_decay_g)} 个参数到优化器。  lr: {lr_g}")

    # # ✅ 在每次解冻完之后冻结 LayerNorm 和 PosEmbed
    # for name, param in vit.named_parameters():
    #     if "norm" in name or "pos_embed" in name:
    #         param.requires_grad = False

    # 🔥 如果添加了新参数组，重建scheduler
    if rebuild_scheduler:
        print(f"🔄 重建Scheduler以包含新参数组...")
        # 重新创建scheduler，它会包含所有当前的参数组
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=1,
            eta_min=5e-6
        )
        print(f"✅ Scheduler已更新，当前管理 {len(optimizer.param_groups)} 个参数组")

    return scheduler

def progressive_unfreeze_convnext_large_with_scheduler(model, optimizer, epoch, scheduler=None):
    """
    双路（object + global）ConvNeXt-Large 渐进式解冻策略
    features[0] = backbone.features(8段), features[1] = avgpool  ← 你的封装方式
    解冻计划：
      epoch 2 : 解冻 stage4(features[7]) + trans4(features[6])
      epoch 6 : 解冻 stage3(features[5]) 最后 6 个 CNBlock
      epoch 10: 再额外解冻 stage3（总共 12 个 CNBlock）
    """
    rebuild_scheduler = False

    # ① 已在 optimizer 中的参数集合（用 id 判断）
    existing = {id(p) for g in optimizer.param_groups for p in g["params"]}

    def split_decay_groups(module):
        decay, no_decay = [], []
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            lname = name.lower()
            if any(k in lname for k in ["bias", "norm", "layernorm", "bn", "ln"]):
                no_decay.append(p)
            else:
                decay.append(p)
        return decay, no_decay

    # ② 只解冻“末尾若干块”
    def unfreeze_last_blocks(stage_module, num_blocks):
        blocks = list(stage_module.children())
        num = min(num_blocks, len(blocks))
        for blk in blocks[-num:]:
            for p in blk.parameters():
                p.requires_grad = True

    # ③ 过滤掉“已经在 optimizer 里的参数”
    def filter_new_params(params):
        return [p for p in params if id(p) not in existing]

    # 遍历 object / global 两路
    for extractor_name in ["object_extractor", "global_extractor"]:
        extractor = getattr(model, extractor_name, None)
        if extractor is None:
            continue
        if "convnext" not in extractor.backbone_name:
            continue

        feats_outer = extractor.features
        feats = feats_outer[0] if len(feats_outer) == 2 else feats_outer  # 你的封装：features = [backbone.features, avgpool]
        if len(feats) < 8:
            print(f"[WARN] {extractor_name}: unexpected features length={len(feats)}; skip unfreeze.")
            continue

        stage3, stage4, trans4 = feats[5], feats[7], feats[6]
        base_lr = 5e-5 if extractor_name == "object_extractor" else 2.5e-5  # global = 1/2

        if epoch == 2:
            # 解冻 stage4 + trans4
            for p in stage4.parameters(): p.requires_grad = True
            for p in trans4.parameters(): p.requires_grad = True

            decay, no_decay = split_decay_groups(stage4)
            d2, nd2 = split_decay_groups(trans4)
            decay += d2; no_decay += nd2

            # ★ 关键：过滤掉已存在的参数，避免重复加入
            decay     = filter_new_params(decay)
            no_decay  = filter_new_params(no_decay)

            if decay or no_decay:
                optimizer.add_param_group({
                    "params": decay,
                    "lr": base_lr,
                    "weight_decay": 0.05,
                    "name": f"{extractor_name}_stage4_decay"
                })
                optimizer.add_param_group({
                    "params": no_decay,
                    "lr": base_lr,
                    "weight_decay": 0.00,
                    "name": f"{extractor_name}_stage4_nodecay"
                })
                print(f"🧩 [{extractor_name}] 解冻 Stage4 + Transition4（新增 {len(decay)+len(no_decay)} 个参数）")
                rebuild_scheduler = True
                # 更新 existing 集合，防止后续重复
                existing.update(id(p) for p in decay)
                existing.update(id(p) for p in no_decay)

        elif epoch == 6:
            unfreeze_last_blocks(stage3, num_blocks=6)
            decay, no_decay = split_decay_groups(stage3)

            decay    = filter_new_params(decay)
            no_decay = filter_new_params(no_decay)

            if decay or no_decay:
                lr = base_lr * 1.6
                optimizer.add_param_group({
                    "params": decay,
                    "lr": lr,
                    "weight_decay": 0.05,
                    "name": f"{extractor_name}_stage3_tail6_decay"
                })
                optimizer.add_param_group({
                    "params": no_decay,
                    "lr": lr,
                    "weight_decay": 0.00,
                    "name": f"{extractor_name}_stage3_tail6_nodecay"
                })
                print(f"🧩 [{extractor_name}] 解冻 Stage3 最后 6 个 CNBlock（新增 {len(decay)+len(no_decay)} 个参数）")
                rebuild_scheduler = True
                existing.update(id(p) for p in decay)
                existing.update(id(p) for p in no_decay)

        elif epoch == 10:
            unfreeze_last_blocks(stage3, num_blocks=12)  # 会覆盖到之前的 6 个，因此必须再过滤一次
            decay, no_decay = split_decay_groups(stage3)

            decay    = filter_new_params(decay)
            no_decay = filter_new_params(no_decay)

            if decay or no_decay:
                lr = base_lr * 2.0
                optimizer.add_param_group({
                    "params": decay,
                    "lr": lr,
                    "weight_decay": 0.05,
                    "name": f"{extractor_name}_stage3_more_decay"
                })
                optimizer.add_param_group({
                    "params": no_decay,
                    "lr": lr,
                    "weight_decay": 0.00,
                    "name": f"{extractor_name}_stage3_more_nodecay"
                })
                print(f"🧩 [{extractor_name}] 解冻 Stage3 额外 12 个 CNBlock（新增 {len(decay)+len(no_decay)} 个参数）")
                rebuild_scheduler = True
                existing.update(id(p) for p in decay)
                existing.update(id(p) for p in no_decay)

    if rebuild_scheduler:
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=5e-6)
        print(f"✅ Scheduler已更新，当前管理 {len(optimizer.param_groups)} 个参数组")
    return scheduler


def grad_norm(model, pattern=None):
    import math, re
    total = 0.0
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        if pattern and not re.search(pattern, n):
            continue
        param_norm = p.grad.data.norm(2).item()
        total += param_norm ** 2
    return total ** 0.5

def train(
        model,
        criterion,
        train_dataloader,
        test_dataloader,
        num_epochs,
        num_classes=5,
        ds_id=None,
        backbone=None,
        attn=None,
        save_results=None
):

    best_f1 = 0.0
    best_result = None
    optimizer = get_optimizer(model = model) # 学习率不变

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=5e-6)

    for epoch in range(num_epochs):
        model.train()
        # 记录各任务预测与标签
        all_preds1, all_labels1 = [], []
        all_preds2, all_labels2 = [], []
        running_loss = 0.0

        # progressive_unfreeze_vit(model, optimizer, epoch)
        if "vit" in model.object_extractor.backbone_name:
            progressive_unfreeze_vit_with_scheduler(
                model,
                optimizer,
                epoch,
                scheduler
            )
        elif "convnext" in model.object_extractor.backbone_name:
            progressive_unfreeze_convnext_large_with_scheduler(
                model,
                optimizer,
                epoch,
                scheduler
            )

        # unfreeze_backbone_gradually(model=model, epoch=epoch, total_epochs=num_epochs)

        # optimizer = get_optimizer(model=model, epoch=epoch, total_epochs=num_epochs)  # 学习率随进程变化
        # optimizer, scheduler = get_optimizer_scheduler(model=model, train_dataloader=train_dataloader, total_epochs=num_epochs)

        for batch_idx, batch in enumerate(train_dataloader):
            imgs_parent = batch["original"].cuda()
            imgs = batch["crop"].cuda();
            y1, y2 = batch["label"];
            y1, y2 = y1.cuda(), y2.cuda()
            bboxes = batch["bbox"].cuda();

            optimizer.zero_grad()

            out1, out2 = model(
                img_global = imgs_parent,
                img_object = imgs,
                labels = [y1, y2],
                bbox = bboxes
            )


            if attn == "bicross":
                loss, loss1, loss2 = criterion((out1, out2), (y1, y2))
                if hasattr(model.attn, "contrastive_reg"):
                    loss += model.attn.contrastive_reg
                if hasattr(model.attn, "loss_gate_reg"):
                    loss += model.attn.loss_gate_reg
                if hasattr(model, "decor_loss"):
                    loss += model.decor_loss
                # model.attn.loss_gate_reg + model.attn.contrastive_reg
            else:
                loss, loss1, loss2 = criterion((out1, out2), (y1, y2))

            loss.backward()

            # ===== 新增：打印梯度范数对比 =====
            raw_grad_all = grad_norm(model)
            raw_grad_head = grad_norm(model, 'classifier1|classifier2|attn')

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            clipped_grad_all = grad_norm(model)
            clipped_grad_head = grad_norm(model, 'classifier1|classifier2|attn')

            if batch_idx == len(train_dataloader) - 1:  # 每个 epoch 打印一次（最后一个 batch）
                print(f"Epoch {epoch + 1}: raw_grad(all)={raw_grad_all:.3f}, raw_grad(head)={raw_grad_head:.3f} | "
                      f"clipped_grad(all)={clipped_grad_all:.3f}, clipped_grad(head)={clipped_grad_head:.3f}")
            # =====================================

            optimizer.step()
            # 🔥 每个batch更新lr
            scheduler.step(epoch + batch_idx / len(train_dataloader))

            running_loss += loss.item() * imgs.size(0)

            preds1 = out1.argmax(dim=1)
            preds2 = out2.argmax(dim=1)
            all_preds1.extend(preds1.cpu().numpy())
            all_preds2.extend(preds2.cpu().numpy())
            all_labels1.extend(y1.cpu().numpy())
            all_labels2.extend(y2.cpu().numpy())

        results = evaluate_multitask(
            all_labels1, all_preds1,
            all_labels2, all_preds2,
            num_classes_1=num_classes[0],
            num_classes_2=num_classes[1]
        )
        print(f"\n🎯 Epoch {epoch + 1}/{num_epochs}")
        print(f"Loss: {running_loss / len(train_dataloader.dataset):.4f}")
        print(f"[Task1] Precision={results['task1']['precision']:.4f}, "
              f"Recall={results['task1']['recall']:.4f}, F1={results['task1']['f1']:.4f}")
        print(f"[Task2] Precision={results['task2']['precision']:.4f}, "
              f"Recall={results['task2']['recall']:.4f}, F1={results['task2']['f1']:.4f}")
        print(f"[Combined] Precision={results['combined']['precision']:.4f}, "
              f"Recall={results['combined']['recall']:.4f}, F1={results['combined']['f1']:.4f}")

        print()
        print(f"---> Epoch {epoch+1}: grad_norm(all)={grad_norm(model):.3f}, "
              f"grad_norm(head)={grad_norm(model, 'classifier1|classifier2|attn'):.3f}")
        print("---> Learning Rate & Param Group:")
        for i, param_group in enumerate(optimizer.param_groups):
            print(f"    Param group {i}: lr = {param_group['lr']}")
        print()

        # 每个epoch后进行测试
        Test_Precision, Test_Recall, Test_F1 = test(model, test_dataloader, num_classes)

        test_f1_value = float(Test_F1.split("±")[0])
        # 比较结果：如果结果更好，保存模型权重+记录结果
        if test_f1_value > best_f1:
            best_f1 = test_f1_value

            model_path = f"./config_file/models/{ds_id}_{backbone}_{attn}_model_multiClassify.pth"
            torch.save(model.state_dict(), model_path)

            best_result = pd.DataFrame({
                "dataset": [ds_id],
                "backbone": [backbone],
                "attn": [attn],
                "precision": [Test_Precision],
                "recall": [Test_Recall],
                "F1": [Test_F1],
                "model_weight_path": [model_path]
            })

            # log_best_f1 = Test_F1
            # log_best_recall = Test_Recall
            # log_best_precision = Test_Precision

    if best_result is not None:
        save_results = pd.concat([save_results, best_result], ignore_index=True)

        print(f"   ✅ Best model saved: {best_result.iloc[0]['model_weight_path']}")
        print(f"   Precision = {best_result.iloc[0]['precision']}")
        print(f"   Recall = {best_result.iloc[0]['recall']}")
        print(f"   F1 = {best_result.iloc[0]['F1']}")

    return save_results


def bootstrap_confidence_interval(metric_fn, y_true, y_pred, num_bootstrap=1000, alpha=0.05):
    """Bootstrap method to estimate confidence interval for a metric."""
    n = len(y_true)
    metrics = []
    rng = np.random.default_rng(42)
    for _ in trange(num_bootstrap, desc="Bootstrapping", leave=False):
        indices = rng.choice(n, n, replace=True)
        sample_true = np.array(y_true)[indices]
        sample_pred = np.array(y_pred)[indices]
        metric_value = metric_fn(sample_true, sample_pred)
        metrics.append(metric_value)
    lower = np.percentile(metrics, 100 * alpha / 2)
    upper = np.percentile(metrics, 100 * (1 - alpha / 2))
    mean = np.mean(metrics)
    half_interval = (upper - lower) / 2
    return mean, half_interval


def test(model, dataloader, num_classes=[3,3], num_bootstrap=1000):  # , class_names
    model.eval()

    all_preds1, all_labels1 = [], []
    all_preds2, all_labels2 = [], []

    with torch.no_grad():
        for batch in dataloader:
            imgs_parent = batch["original"].cuda()
            imgs = batch["crop"].cuda();
            y1, y2 = batch["label"];
            y1, y2 = y1.cuda(), y2.cuda()
            bboxes = batch["bbox"].cuda();

            out1, out2 = model(
                img_global = imgs_parent,
                img_object = imgs,
                bbox = bboxes
            )
            preds1 = out1.argmax(dim=1)
            preds2 = out2.argmax(dim=1)

            all_preds1.extend(preds1.cpu().numpy())
            all_preds2.extend(preds2.cpu().numpy())
            all_labels1.extend(y1.cpu().numpy())
            all_labels2.extend(y2.cpu().numpy())

        ## 测试Results的结果
        # results = evaluate_multitask(
        #     all_labels1, all_preds1,
        #     all_labels2, all_preds2,
        #     num_classes_1=num_classes[0],
        #     num_classes_2=num_classes[1]
        # )
        # print("📊 Evaluation Summary:")
        # print(f"[Task1] Precision={results['task1']['precision']:.4f}, "
        #       f"Recall={results['task1']['recall']:.4f}, F1={results['task1']['f1']:.4f}")
        # print(f"[Task2] Precision={results['task2']['precision']:.4f}, "
        #       f"Recall={results['task2']['recall']:.4f}, F1={results['task2']['f1']:.4f}")
        # print(f"[Combined] Precision={results['combined']['precision']:.4f}, "
        #       f"Recall={results['combined']['recall']:.4f}, F1={results['combined']['f1']:.4f}")
        # print("-------------------------------------------------------")

        # 定义 metric 函数
        def precision_fn(y_true, y_pred):
            p, _, _ = calculate_metrics(y_true, y_pred, average="macro")
            return p

        def recall_fn(y_true, y_pred):
            _, r, _ = calculate_metrics(y_true, y_pred, average="macro")
            return r

        def f1_fn(y_true, y_pred):
            _, _, f = calculate_metrics(y_true, y_pred, average="macro")
            return f

        # 任务1
        p1, p1_ci = bootstrap_confidence_interval(precision_fn, all_labels1, all_preds1, num_bootstrap)
        r1, r1_ci = bootstrap_confidence_interval(recall_fn, all_labels1, all_preds1, num_bootstrap)
        f1, f1_ci = bootstrap_confidence_interval(f1_fn, all_labels1, all_preds1, num_bootstrap)

        # 任务2
        p2, p2_ci = bootstrap_confidence_interval(precision_fn, all_labels2, all_preds2, num_bootstrap)
        r2, r2_ci = bootstrap_confidence_interval(recall_fn, all_labels2, all_preds2, num_bootstrap)
        f2, f2_ci = bootstrap_confidence_interval(f1_fn, all_labels2, all_preds2, num_bootstrap)

        # 合并任务 (将两任务联合为9类标签)
        combined_true = np.array(all_labels1) * num_classes[1] + np.array(all_labels2)
        combined_pred = np.array(all_preds1) * num_classes[1] + np.array(all_preds2)

        pC, pC_ci = bootstrap_confidence_interval(precision_fn, combined_true, combined_pred, num_bootstrap)
        rC, rC_ci = bootstrap_confidence_interval(recall_fn, combined_true, combined_pred, num_bootstrap)
        fC, fC_ci = bootstrap_confidence_interval(f1_fn, combined_true, combined_pred, num_bootstrap)

        # === 打印输出 ===
        print("📊 Testing Log with Bootstrap CIs:")
        print(
            f"[Task1] Precision: {p1:.4f} ± {p1_ci:.4f} | Recall: {r1:.4f} ± {r1_ci:.4f} | F1: {f1:.4f} ± {f1_ci:.4f}")
        print(
            f"[Task2] Precision: {p2:.4f} ± {p2_ci:.4f} | Recall: {r2:.4f} ± {r2_ci:.4f} | F1: {f2:.4f} ± {f2_ci:.4f}")
        print(
            f"[Combined] Precision: {pC:.4f} ± {pC_ci:.4f} | Recall: {rC:.4f} ± {rC_ci:.4f} | F1: {fC:.4f} ± {fC_ci:.4f}")
        print("-------------------------------------------------------------------------")

        # === 返回主任务指标 ===
        Test_Precision = f"{pC:.4f} ± {pC_ci:.4f}"
        Test_Recall = f"{rC:.4f} ± {rC_ci:.4f}"
        Test_F1 = f"{fC:.4f} ± {fC_ci:.4f}"
        return Test_Precision, Test_Recall, Test_F1