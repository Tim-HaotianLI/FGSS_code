import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FeatureExtractor(nn.Module):
    def __init__(self, backbone_name="resnet", pretrained=True):
        super().__init__()
        self.backbone_name = backbone_name.lower()

        if "resnet" in self.backbone_name:
            backbone = getattr(models, self.backbone_name)(
                weights=models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            )
            self.out_features = backbone.fc.in_features
            self.features = nn.Sequential(*list(backbone.children())[:-1])

        elif "convnext" in self.backbone_name:
            backbone = getattr(models, self.backbone_name)(
                weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.out_features = backbone.classifier[2].in_features
            self.features = nn.Sequential(*list(backbone.children())[:-1])

        elif "swin" in self.backbone_name:
            backbone = getattr(models, self.backbone_name)(
                weights=models.Swin_V2_B_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
            self.features = backbone

        elif "vit" in self.backbone_name:
            backbone = getattr(models, self.backbone_name)(
                weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            )
            # print(backbone)
            self.out_features = backbone.heads.head.in_features
            backbone.heads.head = nn.Identity()
            self.features = backbone

    def forward(self, x):
        x = self.features(x)
        if len(x.shape) > 2:
            x = torch.flatten(x, 1);
        # print("x_shape:",x.shape)
        return x


class ElementWiseAttention(nn.Module):
    def __init__(self, feature_dim, use_residual=False):
        super().__init__();
        self.use_residual = use_residual
        self.attn = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # 关键修改
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, f_global, f_object):
        gate = self.attn(torch.cat([f_global, f_object], dim=1))
        f_obj_attn = f_object * gate
        if self.use_residual:
            f_obj_attn = f_obj_attn + f_object
        return f_obj_attn


class CrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=2, use_residual=True):
        super().__init__();
        self.use_residual = use_residual

        self.mha = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.3, # 0.1
            batch_first=True
        )

        # 添加FFN层，增加特征转换
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim * 2, feature_dim),
            # nn.Dropout(0.1)
        )

        self.ln1 = nn.LayerNorm(feature_dim)
        self.ln2 = nn.LayerNorm(feature_dim)
        self.attn_dropout = nn.Dropout(0.3)
        self.dropout = nn.Dropout(0.2) # <==== 这个效果非常好

    def forward(self, f_global, f_object):
        f_global = F.normalize(f_global, p=2, dim=1)
        f_object = F.normalize(f_object, p=2, dim=1)
        # f_global, f_object: [B, C]
        q = f_object.unsqueeze(1)  # [B, 1, C]
        k = f_global.unsqueeze(1)  # [B, 1, C]
        v = f_global.unsqueeze(1)  # [B, 1, C]

        attn_out, _ = self.mha(q, k, v)  # [B, 1, C]
        attn_out = attn_out.squeeze(1)
        attn_out = self.attn_dropout(attn_out)

        if self.use_residual:
            x = self.ln1(attn_out + f_object)
        else:
            x = self.ln1(attn_out)

        # FFN + Residual
        ffn_out = self.ffn(x)
        out = self.dropout(self.ln2(ffn_out + x))
        # out = self.ln2(ffn_out + x)

        return out

class BiCrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=4, depth = 3, dropout=0.1, use_pos_emb=False):
        super().__init__()
        self.use_pos_emb = use_pos_emb
        self.num_heads = num_heads
        self.feature_dim = feature_dim

        self.gate = nn.Parameter(torch.zeros(feature_dim))

        # 可选位置编码（将bbox位置注入注意力）
        if self.use_pos_emb:
            self.pos_fc = nn.Sequential(
                nn.Linear(8, feature_dim),  # sin+cos(4维)=8维
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, feature_dim)
            )

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self_obj": nn.MultiheadAttention(feature_dim, num_heads, dropout, batch_first=True),
                "self_glb": nn.MultiheadAttention(feature_dim, num_heads, dropout, batch_first=True),
                "cross_obj": nn.MultiheadAttention(feature_dim, num_heads, dropout, batch_first=True),
                "cross_glb": nn.MultiheadAttention(feature_dim, num_heads, dropout, batch_first=True),
                "ffn": nn.Sequential(
                    nn.Linear(feature_dim, feature_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(feature_dim * 4, feature_dim)
                ),
                "ln_obj": nn.LayerNorm(feature_dim),
                "ln_glb": nn.LayerNorm(feature_dim)
            }) for _ in range(depth)
        ])

        # # 前馈网络 (FeedForward)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Dropout(dropout)
        )
        self.norm_out = nn.LayerNorm(feature_dim)
    def forward(self, f_global, f_object, bbox=None):
        f_object = F.normalize(f_object, p=2, dim=1)
        f_global = F.normalize(f_global, p=2, dim=1)

        f_object = f_object.unsqueeze(1)
        f_global = f_global.unsqueeze(1)

        # ========== Step 1: 可选位置编码 ==========
        if self.use_pos_emb and bbox is not None:
            bbox = torch.clamp(bbox, 0., 1.)
            sin_emb = torch.sin(bbox * math.pi)
            cos_emb = torch.cos(bbox * math.pi)
            pos_emb = torch.cat([sin_emb, cos_emb], dim=-1)  # [B,8]
            pos_emb = self.pos_fc(pos_emb).unsqueeze(1)

            f_object = f_object + pos_emb
            f_global = f_global + pos_emb

        for l in self.layers:
            # 自注意力
            o_self, _ = l["self_obj"](f_object, f_object, f_object)
            g_self, _ = l["self_glb"](f_global, f_global, f_global)
            # 残差正则化
            f_object = l["ln_obj"](f_object + o_self)
            f_global = l["ln_glb"](f_global + g_self)

            # 交叉注意力
            o_cross, _ = l["cross_obj"](f_object, f_global, f_global)
            g_cross, _ = l["cross_glb"](f_global, f_object, f_object)
            # 残差正则化
            f_object = l["ln_obj"](f_object + o_cross)
            f_global = l["ln_glb"](f_global + g_cross)

            f_object = l["ln_obj"](f_object + l["ffn"](f_object))
            f_global = l["ln_glb"](f_global + l["ffn"](f_global))

            # f_object = f_object / (f_object.norm(dim=-1, keepdim=True) + 1e-6)
            # f_global = f_global / (f_global.norm(dim=-1, keepdim=True) + 1e-6)

        f_object = f_object.mean(dim=1) # 相当于 f_object.unsqueeze(1)
        f_global = f_global.mean(dim=1) # 相当于 f_global.unsqueeze(1)

        # ========== Step 3: gate 融合 ==========
        gate = torch.sigmoid(self.gate).to(f_object.device)  # 保证与输入在同一设备
        self.loss_gate_reg = 0.001 * (gate * (1 - gate)).mean()

        fused = gate * f_object + (1 - gate) * f_global  # 广播到 [B, 1, D]
        self.contrastive_reg = 0.02 * (1 - F.cosine_similarity(fused, f_object, dim=-1)).mean()
        # ========== Step 4: FFN ==========

        fused = fused + self.ffn(self.norm_out(fused))
        # fused = fused + self.ffn(fused)

        return fused  # 输出特征 [B, D]

class CosFaceHead(nn.Module):
    def __init__(self, feature_dim, num_classes, s=20.0, m=0.25):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, x, labels=None):
        x_norm = F.normalize(x, dim=1, eps=1e-6)
        w_norm = F.normalize(self.weight, dim=1, eps=1e-6)
        cosine = torch.matmul(x_norm, w_norm.T)
        if labels is not None:
            one_hot = F.one_hot(labels, num_classes=w_norm.size(0)).float()
            cosine = cosine - one_hot * self.m
        return cosine * self.s


class MinimalClassifier(nn.Module):
    """极简分类器 - 专为小数据集设计"""

    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
            # nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.net(x)

class EnhancedClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.branch2 = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(384, num_classes)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        out = torch.cat([b1, b2], dim=1)
        return self.classifier(out)

class ObjectStateNet(nn.Module):
    def __init__(self,
                 backbone_name,
                 num_classes,
                 pretrained=True,
                 attn_type="element",
                 use_residual=False,
                 fusion="object"
                 ):
        super().__init__()
        self.attn_type = attn_type
        self.object_extractor = FeatureExtractor(backbone_name=backbone_name, pretrained=pretrained)
        self.feature_dim = self.object_extractor.out_features

        # 1. 定义Attention模块
        if self.attn_type == "element":
            self.global_extractor = FeatureExtractor(backbone_name=backbone_name, pretrained=pretrained)
            self.attn = ElementWiseAttention(self.feature_dim, use_residual=use_residual)
            attn_out_dim = self.feature_dim
        elif self.attn_type == "cross":
            self.global_extractor = FeatureExtractor(backbone_name=backbone_name, pretrained=pretrained)
            self.attn = CrossAttention(self.feature_dim,num_heads=2, use_residual=use_residual)
            attn_out_dim = self.feature_dim
        elif self.attn_type == "bicross":
            self.global_extractor = FeatureExtractor(backbone_name=backbone_name, pretrained=pretrained)
            self.attn = BiCrossAttention(self.feature_dim, num_heads=4, depth=2, dropout=0.3, use_pos_emb=True)
            attn_out_dim = self.feature_dim
        elif self.attn_type == "None":
            attn_out_dim = self.feature_dim

        # 2. 定义最终分类头
        self.fusion = fusion
        if self.fusion == "object":
            classifier_in = attn_out_dim
            self.fuse_mlp = None
        elif self.fusion == "concat":
            # 一个小型降维器将 concat -> feature_dim
            self.fuse_mlp = nn.Sequential(
                nn.Linear(attn_out_dim + self.feature_dim, self.feature_dim),
                nn.ReLU(inplace=True)
            )
            classifier_in = self.feature_dim

        self.classifier = CosFaceHead(classifier_in, num_classes)

        self.class_bias = nn.Parameter(torch.zeros(num_classes, self.feature_dim))

        # self.rot_proj = nn.Linear(self.feature_dim, self.feature_dim, bias=False)
        # nn.init.orthogonal_(self.rot_proj.weight)

    def forward(self, img_global, img_object, labels=None, bbox=None):

        f_object = self.object_extractor(img_object)

        # 计算被global调制的object表示

        if self.attn_type != "None":
            # print(self.attn_type)
            f_global = self.global_extractor(img_global)
            if self.attn_type == "bicross":
                f_object = self.attn(f_global = f_global,f_object = f_object,bbox = bbox)
            elif self.attn_type in ["element", "cross"]:
                f_object = self.attn(f_global = f_global, f_object = f_object)

        if self.fusion == "object":
            feat = f_object
        elif self.fusion == "concat":
            feat = torch.cat([f_object, f_global], dim=1)
            feat = self.fuse_mlp(feat)

        # ✅ === Feature Regularization ===
        # if self.training:  # 只在训练阶段启用
        #     f_feat = F.normalize(feat, dim=1)
        #     noise = 0.1 * torch.randn_like(feat) * (1 - f_feat ** 2)
        #     feat = feat + noise

        if self.training:
            feat_plus = feat + self.class_bias[labels]  # ← 需要把 labels 传入 forward，或在 train loop 里做
            logits = self.classifier(feat_plus)
        else:
            # 推理时：用最近原型近似选择偏置（不改接口也行，做成可选）
            with torch.no_grad():
                proto = self.class_bias  # [C,D]
                sim = F.cosine_similarity(feat.unsqueeze(1), proto.unsqueeze(0), dim=-1)  # [B,C]
                idx = sim.argmax(dim=1)
            feat_plus = feat + self.class_bias[idx]
            logits = self.classifier(feat_plus)

        # ✅ === Feature Decorrelation Regularization ===
        if self.training:
            f_obj = F.normalize(f_object, dim=1)  # 只针对 object extractor 的输出
            cov = f_obj.T @ f_obj / f_obj.size(0)
            I = torch.eye(cov.size(0), device=cov.device)
            decor_loss = ((cov - I) ** 2).mean()
            self.decor_loss = 0.001 * decor_loss  # 可调 1e-3 ~ 1e-4

        return logits