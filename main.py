from util.config import get_backbone_config
from modules.backbone import FocalNet
from modules.sup_con_model import SupConModel
from util.train_util import train_epoch, configure_optimizers
import torch
from util.logger import build_logger
from util.dataset_loader_wsss import SupConDataset
from util.data_loade_util import TwoCropTransform
from torch.utils.data import DataLoader



def main():
    vis_backbone = FocalNet(**get_backbone_config().focal)
    vis_backbone.load_state_dict(torch.load('pretrained/focalnet_small_lrf.pth')['model'], strict=True)
    model = SupConModel(backbone=vis_backbone, num_classes=3).cuda()
    logger, _ = build_logger('logger')

    train_dataset = SupConDataset(root_dir="C:\\Users\massoud\PycharmProjects\\700perizie_cleaning\wsss_image_annotation_extaction",
                            mode='train',
                            transform=TwoCropTransform())
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    opts_lr_schedulers = configure_optimizers(model, len(train_dataloader))

    for epoch in range(200):
        train_epoch(model, train_dataloader, opts_lr_schedulers, epoch, len(train_dataloader), logger)






if '__main__' == __name__:
    main()