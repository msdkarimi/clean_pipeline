from util.config import get_backbone_config
from modules.backbone import FocalNet
from modules.sup_con_model import SupConModel
from util.train_util import train_epoch, configure_optimizers
import torch
from util.logger import build_logger



def main():
    vis_backbone = FocalNet(**get_backbone_config().focal)
    vis_backbone.load_state_dict(torch.load('pretrained/focalnet_small_lrf.pth')['model'], strict=True)
    model = SupConModel(backbone=vis_backbone, num_classes=3)
    logger = build_logger('logger')

    train_loader = [1]
    opts_lr_schedulers = configure_optimizers(model, len(train_loader))

    for epoch in range(200):
        train_epoch(model, train_loader, opts_lr_schedulers, epoch, len(train_loader), logger)



if '__main__' == __name__:
    main()