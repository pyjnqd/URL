'''
tsa.py
Created by Wei-Hong Li [https://weihonglee.github.io]
This code allows you to attach task-specific parameters, including adapters, pre-classifier alignment (PA) mapping
from 'Universal Representation Learning from Multiple Domains for Few-shot Classification'
(https://arxiv.org/pdf/2103.13841.pdf), to a pretrained backbone. 
It only learns attached task-specific parameters from scratch on the support set to adapt 
the pretrained model for previously unseen task with very few labeled samples.
'Cross-domain Few-shot Learning with Task-specific Adapters.' (https://arxiv.org/pdf/2107.00358.pdf)
'''

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import math
from config import args
import copy
import torch.nn.functional as F
from models.losses import prototype_loss
from utils import device

class conv_tsa(nn.Module):
    def __init__(self, orig_conv):
        super(conv_tsa, self).__init__()
        # the original conv layer
        self.conv = copy.deepcopy(orig_conv)
        self.conv.weight.requires_grad = False
        planes, in_planes, _, _ = self.conv.weight.size()
        stride, _ = self.conv.stride
        # task-specific adapters
        if 'alpha' in args['test.tsa_opt']:
            self.alpha = nn.Parameter(torch.ones(planes, in_planes, 1, 1))
            self.alpha.requires_grad = True

    def forward(self, x):
        y = self.conv(x)
        if 'alpha' in args['test.tsa_opt']:
            # residual adaptation in matrix form
            y = y + F.conv2d(x, self.alpha, stride=self.conv.stride)
        return y

class pa(nn.Module):
    """ 
    pre-classifier alignment (PA) mapping from 'Universal Representation Learning from Multiple Domains for Few-shot Classification'
    (https://arxiv.org/pdf/2103.13841.pdf)
    """
    def __init__(self, feat_dim):
        super(pa, self).__init__()
        # define pre-classifier alignment mapping
        self.weight = nn.Parameter(torch.ones(feat_dim, feat_dim, 1, 1))
        self.weight.requires_grad = True

    def forward(self, x):
        if len(list(x.size())) == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
            x = F.conv2d(x, self.weight.to(x.device)).flatten(1)
        else:
            x = F.conv2d(x, self.weight.to(x.device))
        return x

class resnet_tsa(nn.Module):
    """ Attaching task-specific adapters (alpha) and/or PA (beta) to the ResNet backbone """
    def __init__(self, orig_resnet):
        super(resnet_tsa, self).__init__()
        # freeze the pretrained backbone
        for k, v in orig_resnet.named_parameters():
                v.requires_grad=False

        # attaching task-specific adapters (alpha) to each convolutional layers
        # note that we only attach adapters to residual blocks in the ResNet
        for block in orig_resnet.layer1:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_tsa(m)
                    setattr(block, name, new_conv)

        for block in orig_resnet.layer2:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_tsa(m)
                    setattr(block, name, new_conv)

        for block in orig_resnet.layer3:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_tsa(m)
                    setattr(block, name, new_conv)

        for block in orig_resnet.layer4:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_tsa(m)
                    setattr(block, name, new_conv)

        self.backbone = orig_resnet

        # attach pre-classifier alignment mapping (beta)
        feat_dim = orig_resnet.layer4[-1].bn2.num_features
        beta = pa(feat_dim)
        setattr(self, 'beta', beta)

        # fc_clu = nn.Sequential(
        #     nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 128)
        # )#nn.Linear(512, 128)
        # setattr(self, 'fc_clu', fc_clu)
        # fc_loc = nn.Linear(512, 4)
        # setattr(self, 'fc_loc', fc_loc)


    def forward(self, x):
        return self.backbone.forward(x=x)

    def embed(self, x):
        return self.backbone.embed(x)
    def embed2(self, x):
        return self.backbone.embed2(x)

    def get_state_dict(self):
        """Outputs all the state elements"""
        return self.backbone.state_dict()

    def get_parameters(self):
        """Outputs all the parameters"""
        return [v for k, v in self.backbone.named_parameters()]

    def reset(self):
        # initialize task-specific adapters (alpha)
        for k, v in self.backbone.named_parameters():
            if 'alpha' in k:
                v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device) * 0.0001

        # initialize pre-classifier alignment mapping (beta)
        v = self.beta.weight
        self.beta.weight.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)


def tsa(images_gather, permute, bs_all, context_images, context_labels, model, max_iter=40, lr=0.1, lr_beta=1, distance='cos'):
    """
    Optimizing task-specific parameters attached to the ResNet backbone, 
    e.g. adapters (alpha) and/or pre-classifier alignment mapping (beta)
    """
    model.eval()
    tsa_opt = args['test.tsa_opt']
    alpha_params = [v for k, v in model.named_parameters() if 'alpha' in k]
    beta_params = [v for k, v in model.named_parameters() if 'beta' in k]
    # fc_clu_params = [v for k, v in model.named_parameters() if 'fc_clu' in k]
    # fc_loc_params = [v for k, v in model.named_parameters() if 'fc_loc' in k]
    params = []
    if 'alpha' in tsa_opt:
        params.append({'params': alpha_params})
    if 'beta' in tsa_opt:
        params.append({'params': beta_params, 'lr': lr_beta})
    # params.append({'params': fc_clu_params})
    # params.append({'params': fc_loc_params})
    optimizer = torch.optim.Adadelta(params, lr=lr)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30, 0.01, 25)

    if 'alpha' not in tsa_opt:
        with torch.no_grad():
            context_features = model.embed(context_images)
    # # ======================================= #
    from LightningFSL.modules.Jigsaw import SupCluLoss

    criterion_clu = SupCluLoss(temperature=0.3)
    criterion_loc = nn.CrossEntropyLoss()
    ada_avg_pool2d = nn.AdaptiveAvgPool2d((2, 2))

    # ================================================ #
    for i in range(max_iter):
        optimizer.zero_grad()
        model.zero_grad()

        if 'alpha' in tsa_opt:
            # adapt features by task-specific adapters
            context_features = model.embed(context_images)
        if 'beta' in tsa_opt:
            # adapt feature by PA (beta)
            aligned_features = model.beta(context_features)
        else:
            aligned_features = context_features
        loss1, stat, _ = prototype_loss(aligned_features, context_labels,
                                       aligned_features, context_labels, distance=distance)

        # ============================ #

        from PIL import ImageFilter
        import random
        import torchvision.transforms as T
        from torchvision import transforms

        # class GaussianBlur(object):
        #     """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
        #
        #     def __init__(self, sigma=[.1, 2.]):
        #         self.sigma = sigma
        #
        #     def __call__(self, x):
        #         sigma = random.uniform(self.sigma[0], self.sigma[1])
        #         x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        #         return x
        #
        # transform_patch = transforms.Compose([
        #     transforms.RandomResizedCrop(42, scale=(0.2, 1.0)),
        #     transforms.RandomApply([
        #         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        #     ], p=0.8),
        #     transforms.RandomGrayscale(p=0.2),
        #     transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(np.array([0.485, 0.456, 0.406]),
        #                          np.array([0.229, 0.224, 0.225]))
        # ])
        #
        # to_pil = T.ToPILImage()
        # patches = [[], [], [], []]
        #
        # for j, x in enumerate(concat_images):
        #     x = x.to(torch.uint8)
        #     img = to_pil(x)
        #     h, w = img.size
        #     ch = 0.3 * h
        #     cw = 0.3 * w
        #     one_patches = [transform_patch(img.crop((0, 0, h // 2 + ch, w // 2 + cw))),
        #                    transform_patch(img.crop((0, w // 2 - cw, h // 2 + ch, w))),
        #                    transform_patch(img.crop((h // 2 - ch, 0, h, w // 2 + cw))),
        #                    transform_patch(img.crop((h // 2 - ch, w // 2 - cw, h, w)))]
        #     patches[0].append(one_patches[0])
        #     patches[1].append(one_patches[1])
        #     patches[2].append(one_patches[2])
        #     patches[3].append(one_patches[3])
        # for k in range(4):
        #     patches[k] = torch.stack(patches[k])

        # @torch.no_grad()
        # def _batch_gather(images):
        #     images_gather = images
        #     n, c, h, w = images_gather[0].shape
        #     permute = torch.randperm(n * 4).cuda()
        #     images_gather = torch.cat(images_gather, dim=0)
        #     images_gather = images_gather[permute, :, :, :]
        #     col1 = torch.cat([images_gather[0:n], images_gather[n:2 * n]], dim=3)
        #     col2 = torch.cat([images_gather[2 * n:3 * n], images_gather[3 * n:]], dim=3)
        #     images_gather = torch.cat([col1, col2], dim=2)
        #
        #     return images_gather, permute, n
        #
        # images_gather, permute, bs_all = _batch_gather(patches)



        # adapt features by task-specific adapters
        q = model.embed2(images_gather.cuda())
        # adapt feature by PA (beta)
        # q = model.beta(q)


        q = F.interpolate(q, size=(8,8), mode='bilinear', align_corners=False)
        q = ada_avg_pool2d(q)
        q = model.beta(q)
        q_gather = q

        n, c, h, w = q_gather.shape
        c1, c2 = q_gather.split([1, 1], dim=2)
        f1, f2 = c1.split([1, 1], dim=3)
        f3, f4 = c2.split([1, 1], dim=3)
        q_gather = torch.cat([f1, f2, f3, f4], dim=0)
        q_gather = q_gather.view(n * 4, -1)

        # clustering branch
        # for way-clustering
        way = images_gather.shape[0] // 5
        label_clu_way = torch.LongTensor(
            list(np.reshape([[j] * 5 for j in range(way)], (1, -1)).squeeze()) * 4)
             #list(np.reshape([[j] * 10 for j in range(way)], (1, -1)).squeeze())) * 4 )
        label_clu_way = label_clu_way[permute].cuda()

        # for image-clustering
        #label_clu_image = permute % bs_all


        # q_clu = model.fc_clu(q_gather.cuda())
        q_clu = nn.functional.normalize(q_gather, dim=1)
        loss3, stat, _ = prototype_loss(q_clu, label_clu_way,
                                        q_clu, label_clu_way, distance=distance)

        # location branch
        # label_loc = torch.LongTensor([0] * bs_all + [1] * bs_all + [2] * bs_all + [3] * bs_all).cuda()
        # label_loc = label_loc[permute]
        # q_loc = model.fc_loc(q_gather)

        # loss_clu_way = criterion_clu(q_clu, label_clu_way)
        # loss_clu_image = criterion_clu(q_clu, label_clu_image)
        # loss_loc = criterion_loc(q_loc, label_loc)

        # loss2 = 0.5 * loss_clu_way #+ 0.5 * loss_clu_image

        loss = loss3

        loss.backward()
        optimizer.step()
        #lr_scheduler.step()
    return