import random
import torch.nn.functional as F
from torch import nn

from .DNLPCrowd import ConvMLP
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, VisionTransformer
import torch
from .graph import GraphReasoning

class CGR(nn.Module):
    def __init__(self, n_class=2, n_iter=2, chnn_side=(768, 384, 192), chnn_targ=(768, 192, 48, 768), rd_sc=32, dila=(4, 8, 16)):
        super().__init__()
        self.n_graph = len(chnn_side)
        n_node = len(dila)
        graph = [GraphReasoning(ii, rd_sc, dila, n_iter) for ii in chnn_side]
        self.graph = nn.ModuleList(graph)
        C_cat = [nn.Sequential(
            nn.Conv2d(ii//rd_sc*n_node, ii//rd_sc, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ii//rd_sc),
            nn.ReLU(inplace=True))
            for ii in (chnn_side+chnn_side)]
        self.C_cat = nn.ModuleList(C_cat)
        idx = [ii for ii in range(len(chnn_side))]
        C_up = [nn.Sequential(
            nn.Conv2d(chnn_targ[ii]+chnn_side[ii]//rd_sc, chnn_targ[ii+1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(chnn_targ[ii+1]),
            nn.ReLU(inplace=True))
            for ii in (idx+idx)]
        self.C_up = nn.ModuleList(C_up)
        self.C_cls = nn.Conv2d(chnn_targ[-1]*2, n_class, 1)

    def forward(self, inputs):
        img = inputs
        cas_rgb = img[0]
        nd_rgb, nd_key = None, False
        for ii in range(self.n_graph):
            feat_rgb = self.graph[ii]([img[ii], nd_rgb], nd_key)
            feat_rgb = torch.cat(feat_rgb, 1)
            feat_rgb = self.C_cat[ii](feat_rgb)
            nd_rgb, nd_key = feat_rgb, True
            cas_rgb = torch.cat((feat_rgb, cas_rgb), 1)
            cas_rgb = F.interpolate(cas_rgb, scale_factor=2, mode='bilinear', align_corners=True)
            cas_rgb = self.C_up[ii](cas_rgb)
        return cas_rgb

class DMFmlp_gap(ConvMLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(110592, 128),
            nn.Dropout(0.6),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.output1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(110592, 128),
            # nn.Linear(126720, 128),
            nn.Dropout(0.6),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # self.conv1 = nn.Conv2d(192, 960, kernel_size=4,stride=4)
        self.graph = CGR().cuda()
        checkpoint = torch.load('/home/dengmingfang/下载/TransCrowd-main/Networks/CasGnn.pth')
        self.graph.load_state_dict(checkpoint, strict=False)
        self.output1.apply(self.init_weight)


    def generate_feature_patches(self, unlabel_x, ratio=None):
        # unlabeled image processing

        unlabel_x_1 = unlabel_x.cuda()
        b, c, h, w = unlabel_x.shape

        center_x = random.randint(h // 2 - (h - h * ratio) // 2, h // 2 + (h - h * ratio) // 2)
        center_y = random.randint(w // 2 - (w - w * ratio) // 2, w // 2 + (w - w * ratio) // 2)

        new_h2 = int(h * ratio)
        new_w2 = int(w * ratio)  # 48*48
        unlabel_x_2 = unlabel_x[:, :, center_x - new_h2 // 2:center_x + new_h2 // 2,
                      center_y - new_w2 // 2:center_y + new_w2 // 2]

        new_h3 = int(new_h2 * ratio)
        new_w3 = int(new_w2 * ratio)
        unlabel_x_3 = unlabel_x[:, :, center_x - new_h3 // 2:center_x + new_h3 // 2,
                      center_y - new_w3 // 2:center_y + new_w3 // 2]

        new_h4 = int(new_h3 * ratio)
        new_w4 = int(new_w3 * ratio)
        unlabel_x_4 = unlabel_x[:, :, center_x - new_h4 // 2:center_x + new_h4 // 2,
                      center_y - new_w4 // 2:center_y + new_w4 // 2]

        new_h5 = int(new_h4 * ratio)
        new_w5 = int(new_w4 * ratio)
        unlabel_x_5 = unlabel_x[:, :, center_x - new_h5 // 2:center_x + new_h5 // 2,
                      center_y - new_w5 // 2:center_y + new_w5 // 2]

        unlabel_x_2 = nn.functional.interpolate(unlabel_x_2, size=(h, w), mode='bilinear')
        unlabel_x_3 = nn.functional.interpolate(unlabel_x_3, size=(h, w), mode='bilinear')
        unlabel_x_4 = nn.functional.interpolate(unlabel_x_4, size=(h, w), mode='bilinear')
        unlabel_x_5 = nn.functional.interpolate(unlabel_x_5, size=(h, w), mode='bilinear')

        unlabel_x = torch.cat([unlabel_x_1, unlabel_x_2, unlabel_x_3, unlabel_x_4, unlabel_x_5], dim=0)
        unlabel_x = torch.split(unlabel_x, split_size_or_sections=b, dim=0)

        return unlabel_x

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.tokenizer(x)
        x = self.conv_stages(x)
        temp2 = x
        x = x.permute(0, 2, 3, 1)
        i = 0
        temp0 = 0
        temp1 = 0
        for stage in self.stages:
            i = i + 1
            x = stage(x)
            if i == 1:
                temp1 = x.permute(0, 3, 1, 2)
            if i == 2:
                temp0 = x.permute(0, 3, 1, 2)
        x_g = self.graph((temp0, temp1, temp2))
        x_g = F.interpolate(x_g, scale_factor=0.125, mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
        x = x + x_g
        unlabel_x = self.generate_feature_patches(x, 0.75)
        x = x.contiguous().view(x.shape[0], -1)
        unlabel_x0 = unlabel_x[0].view(B, -1)
        unlabel_x1 = unlabel_x[1].view(B, -1)
        unlabel_x2 = unlabel_x[2].view(B, -1)
        unlabel_x3 = unlabel_x[3].view(B, -1)
        unlabel_x4 = unlabel_x[4].view(B, -1)
        unlabel_x0 = self.output1(unlabel_x0)
        unlabel_x1 = self.output1(unlabel_x1)
        unlabel_x2 = self.output1(unlabel_x2)
        unlabel_x3 = self.output1(unlabel_x3)
        unlabel_x4 = self.output1(unlabel_x4)
        unlabel_x = torch.cat([unlabel_x0, unlabel_x1, unlabel_x2, unlabel_x3, unlabel_x4], dim=0)
        unlabel_x = torch.split(unlabel_x, split_size_or_sections=B, dim=0)
        x = self.output1(x)
        return x, unlabel_x

@register_model
def base_patch4_224_gap(pretrained=False, **kwargs):
    model = DMFmlp_gap(
        blocks= [4, 8, 3], dims=[192, 384, 768], mlp_ratios=[3, 3, 3], channels=96, n_conv_blocks=3,
        classifier_head=None, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        '''download from https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth'''
        checkpoint = torch.load('./Networks/convmlp_l_imagenet.pth')
        model.load_state_dict(checkpoint, strict=False)
        print("load transformer pretrained")
    return model
