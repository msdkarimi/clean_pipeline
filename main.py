from absl.logging import vlog_is_on

from util.config import get_backbone_config
from modules.backbone_copy import FocalNet
from modules.sup_con_model import SupConModel
from util.train_util import train_epoch, configure_optimizers, validation, from_pretrained
import torch
from util.logger import build_logger
from util.dataset_loader import SupConDataset
from util.data_loade_util import TwoCropTransform
from torch.utils.data import DataLoader
import os


def main():
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('plots/embed', exist_ok=True)
    os.makedirs('plots/cm', exist_ok=True)

    logger, _ = build_logger('logger')
    vis_backbone = FocalNet(**get_backbone_config().focal)
    # vis_backbone.load_state_dict(torch.load('pretrained/focalnet_small_lrf.pth')['model'])

    vis_backbone = from_pretrained(model=vis_backbone, load_dir='pretrained/focalnet_small_lrf.pth', logger=logger)

    model = SupConModel(backbone=vis_backbone, num_classes=3).cuda()


    train_dataset = SupConDataset(root_dir="C:\\Users\massoud\PycharmProjects\\700perizie_cleaning\wsss_image_annotation_extaction",
                            mode='train',
                            transform=TwoCropTransform())
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    val_dataset = SupConDataset(root_dir="C:\\Users\massoud\PycharmProjects\\700perizie_cleaning\wsss_image_annotation_extaction",
                            mode='val',
                            transform=TwoCropTransform(mode='val'))
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    opts_lr_schedulers = configure_optimizers(model, len(train_dataloader))

    _best_f1 = -1.
    for epoch in range(200):
        train_epoch(model, train_dataloader, opts_lr_schedulers, epoch, len(train_dataloader), logger)
        f1_score = validation(model, val_dataloader, epoch, logger)
        if f1_score > _best_f1:
            _best_f1 = f1_score
            torch.save(model.state_dict(), f'checkpoints/e_{epoch}_f1_{f1_score}.pth')

if '__main__' == __name__:
    main()