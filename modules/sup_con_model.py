import torch.nn as nn
from util.train_util import AverageMeter

def __init__(self, backbone: nn.Module, num_classes: int,
             sup_con_criterion, train_dataloader,
             val_dataloader, cls_criterion=None,
             embed_dim: int = 768, log_every=50,
             epochs=200):
    super().__init__()
    self.backbone = backbone
    self._cls_head = nn.Linear(in_features=embed_dim, out_features=num_classes)
    self._sup_con_head = nn.Sequential(
        nn.Linear(in_features=embed_dim, out_features=embed_dim // 2),
        nn.ReLU(),
        nn.Linear(in_features=embed_dim // 2, out_features=768),
    )

    self.sup_con_criterion = sup_con_criterion
    self.cls_criterion = cls_criterion
    self.cls_weights = [0.723198, 0.686115, 0.590128]

    self.train_dataloader = train_dataloader
    self.val_dataloader = val_dataloader

    (self.optimizer_sup_con, self.optimizer_cls,
     self.lr_scheduler_sup_con, self.lr_scheduler_cls) = self.configure_optimizers()

    self.loss_meter_supcon = AverageMeter()
    self.loss_meter_cls = AverageMeter()
    self.grad_norm_meter_supCon = AverageMeter()
    self.grad_norm_meter_cls = AverageMeter()

    self.log_every = log_every
    self.epoch = 0
    self.epochs = epochs
    self.step = 0

    self.f_score, self.sil_score = 0., 0.

    self.logger, _run_folder = build_logger(__name__)