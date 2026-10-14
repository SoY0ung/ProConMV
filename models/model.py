import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models

from models.rwkv import VRWKV_Block, CrossVRWKV_Mix


class MVCBM3(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, concept_classes=6, last_channel=512, dim=128, device='cuda:0'):
        super().__init__()
        self.classes = num_classes
        self.concept_classes = concept_classes
        self.lam_raw = nn.Parameter(torch.tensor(0.5))

        # ======== Load and patch VGG16 ========
        vgg = models.vgg16(pretrained=True)
        first_conv = vgg.features[0]
        new_conv = nn.Conv2d(in_chans, 64, kernel_size=3, stride=1, padding=1)

        with torch.no_grad():
            new_conv.weight[:, :3] = first_conv.weight
            if in_chans > 3:
                extra = torch.mean(first_conv.weight, dim=1, keepdim=True)
                new_conv.weight[:, 3:] = extra.repeat(1, in_chans - 3, 1, 1)
            new_conv.bias = first_conv.bias

        vgg.features[0] = new_conv

        # ======== Backbone stages ========
        self.stage123 = nn.Sequential(*vgg.features[:17])  # Stage 1+2+3: Conv2d 3->256
        self.stage4 = VRWKV_Block(n_embd=256, n_layer=1)  # Replaces VGG stage4
        self.conv = nn.Conv2d(256, 512, kernel_size=1)
        self.stage5 = VRWKV_Block(n_embd=512, n_layer=1)  # Replaces VGG stage5
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ======== Concept representation ========
        self.conv11 = nn.Conv2d(last_channel, concept_classes * dim, kernel_size=1)
        self.split_sizes = [dim] * concept_classes
        self.word_vec = torch.from_numpy(np.load('./models/dr_concept_embeddings.npy')).float().to(device)

        self.mids = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(dim, dim),
                nn.Dropout(0.1)
            ) for _ in range(concept_classes)
        ])

        self.concecpt_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 16),
                nn.Dropout(0.1),
                nn.Linear(16, 1)
            ) for _ in range(concept_classes)
        ])

        self.cross_a = CrossVRWKV_Mix(dim=dim, recurrence=2, key_norm=True)

        self.result_classifier = nn.Sequential(
            nn.Linear(concept_classes, 64),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, intervention=None, inter_c=0, return_feat=False):
        if x.dim() == 4:
            x = x.unsqueeze(1)  # [B, 1, C, H, W]

        B, V, C, H, W = x.shape
        x = x.view(B * V, C, H, W)


        feat = self.stage123(x)  # [B*V, 256, 28, 28]
        feat = self.stage4(feat, (32, 32))
        feat = self.pool4(feat)  # [B*V, 256, 14, 14]
        feat = self.stage5(self.conv(feat), (16, 16))
        feat = self.pool5(feat)  # [B*V, 512, 7, 7]

        # ===== Concept representation =====
        feat = self.conv11(feat)  # [B*V, K*dim, 7, 7]
        splits = torch.chunk(feat, self.concept_classes, dim=1)  # list of [B*V, dim, 7, 7]


        concept_mid = [clf(s) for clf, s in zip(self.mids, splits)]  # [B*V, dim] * K


        concept_embed = [self.cross_a(concept_mid[i], self.word_vec[i])
                         for i in range(self.concept_classes)]  # [B*V, dim] * K


        concept_out = [self.concecpt_classifiers[i](concept_embed[i])
                       for i in range(self.concept_classes)]  # [B*V, 1] * K
        concept_out = torch.cat(concept_out, dim=1)  # [B*V, K]


        concepts_all = concept_out.view(B, V, self.concept_classes)  # [B, V, K]
        feat_for_dr = concepts_all.reshape(B * V, -1)  # [B*V, K]
        res_outputs = self.result_classifier(concepts_all.reshape(B * V, -1))  # [B*V, num_classes]
        res_outputs = res_outputs.view(B, V, self.classes)  # [B, V, num_classes]


        e_p = F.softplus(res_outputs)
        alpha_p = e_p + 1
        alpha0_p = alpha_p.sum(dim=-1)
        conf_pred = 1.0 - (self.classes / (alpha0_p + 1e-8))  # [B, V]

        e_c = F.softplus(concepts_all)
        alpha_c = e_c + 1
        alpha0_c = alpha_c.sum(dim=-1)
        conf_conc = 1.0 - (self.concept_classes / (alpha0_c + 1e-8))  # [B, V]

        lam = torch.sigmoid(self.lam_raw)
        s = lam * conf_conc + (1 - lam) * conf_pred
        w = F.softmax(s, dim=1)  # [B, V]

        fused_preds = (w.unsqueeze(-1) * res_outputs).sum(dim=1)  # [B, num_classes]

        if return_feat:
            return concepts_all, fused_preds, feat_for_dr
        else:
            return concepts_all, fused_preds
