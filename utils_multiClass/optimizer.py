
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR


# 改进优化器的配置
# 改进优化器的配置
def get_optimizer(model, epoch=None, total_epochs=None):
    """根据训练阶段，动态调整学习率"""
    params = []

    if epoch != None and total_epochs != None:
        progress = epoch / total_epochs

        if progress <= 0.3:  # 早期：高学习率快速收敛
            backbone_lr = 0
            attn_lr = 1e-5
            classifier_lr = 5e-5
            # attn_lr = 1e-4
            # classifier_lr = 3e-4
        elif progress <= 0.6:  # 中期：降低学习率
            backbone_lr = 1e-7
            attn_lr = 1e-5
            classifier_lr = 5e-5
            # attn_lr = 5e-5
            # classifier_lr = 1e-4
        else:  # 后期：小学习率精调
            backbone_lr = 1e-8
            attn_lr = 1e-5
            classifier_lr = 3e-5
    elif epoch == None and total_epochs == None:
        # 默认使用较高的学习率
        # backbone_lr = 1e-4
        attn_lr = 1e-5
        classifier_lr = 5e-5

    # backbone 参数 (如果解冻)
    # backbone_params = []
    # for name, param in model.object_extractor.named_parameters():
    #     if param.requires_grad:
    #         # print("local extractor need gradient")
    #         backbone_params.append(param)
    # if hasattr(model, 'global_extractor'):
    #     for name, param in model.global_extractor.named_parameters():
    #         if param.requires_grad:
    #             # print("global extractor need gradient")
    #             backbone_params.append(param)
    #
    # if len(backbone_params) > 0:
    #     params.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": 0.05})  # 很小的学习率

    ## Attention 层
    if hasattr(model, 'attn'):
        params.append({"params": model.attn.parameters(), "lr": attn_lr, "weight_decay": 0.05})

    ## 分类器层
    params.append({"params": model.classifier1.parameters(), "lr": classifier_lr, "weight_decay": 0.01})
    params.append({"params": model.classifier2.parameters(), "lr": classifier_lr, "weight_decay": 0.01})

    optimizer = optim.AdamW(params=params,  betas=(0.9, 0.999))  # 使用AdamW + weight decay # weight_decay=0.05,
    return optimizer

## 添加了仿射变换作为数据增强 + self.gate = nn.Parameter(torch.zeros(feature_dim).fill_(0.5)) 【门控换成这个了】