from util.config import get_backbone_config
from modules.backbone import FocalNet
from modules.sup_con_model import SupConModel
from util.train_util import train_epoch, configure_optimizers

import torch

def main():
    vis_backbone = FocalNet(**get_backbone_config().focal).load_state_dict(torch.load('pretrained/focalnet_small_lrf.pth')['model'])
    model = SupConModel(backbone=vis_backbone, num_classes=3)

    train_loader = []

    opts_lr_schedulers = configure_optimizers(model, len(train_loader))

    for epoch in range(200):
        train_epoch(model, train_loader, )










if '__main__' == __name__:
    main()