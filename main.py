from util.config import get_backbone_config
from modules.backbone import FocalNet
from modules.sup_con_model import SupConModel
import torch

def main(_args):
    vis_backbone = FocalNet(**get_backbone_config().focal).load_state_dict(torch.load('pretrained/focalnet_small_lrf.pth')['model'])
    model = SupConModel(backbone=vis_backbone, num_classes=3)









if '__main__' == __name__:
    main()