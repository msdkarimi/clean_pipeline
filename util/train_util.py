import torch
from util.optimizer import build_optimizer
from util.lr_scheduler import build_scheduler
from util.config import get_opt_config, get_lr_scheduler_config
from util.loss import SupConLoss
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def configure_optimizers(model, num_training_steps_per_epoch):
    optimizer_sup_con = build_optimizer(get_opt_config('sup_con'), [model.backbone, model._sup_con_head])
    optimizer_cls = build_optimizer(get_opt_config('cls'), model._cls_head)


    lr_scheduler_sup_con = build_scheduler(get_lr_scheduler_config(), optimizer_sup_con, num_training_steps_per_epoch)
    lr_scheduler_cls = build_scheduler(get_lr_scheduler_config(), optimizer_cls, num_training_steps_per_epoch)

    return (
        optimizer_sup_con, optimizer_cls , lr_scheduler_sup_con, lr_scheduler_cls
    )

def criterion(sup_con_logits, cls_logits, labels, cls_weights=None):
    sup_con = SupConLoss()
    sup_con_loss = sup_con(sup_con_logits, labels)
    cls_loss = nn.CrossEntropyLoss(weight=cls_weights)(cls_logits, labels)
    return sup_con_loss, cls_loss

def take_step(model, sup_con_loss, cls_loss, batch_idx, epoch, steps_per_epoch, opts_lr_schedulers, grad_norm_meter_supCon, grad_norm_meter_cls):
    optimizer_sup_con, optimizer_cls , lr_scheduler_sup_con, lr_scheduler_cls = opts_lr_schedulers


    optimizer_sup_con.zero_grad()
    sup_con_loss.backward()
    _params = list(model.backbone.parameters()) + list(model._sup_con_head.parameters())
    grad_norm_meter_supCon.update(get_grad_norm(parameters=_params))
    optimizer_sup_con.step()
    lr_scheduler_sup_con.step_update((epoch * steps_per_epoch) + batch_idx)

    optimizer_cls.zero_grad()
    cls_loss.backward()
    _params = list(model._cls_head.parameters())
    grad_norm_meter_cls.update(get_grad_norm(parameters=_params))
    optimizer_cls.step()
    lr_scheduler_cls.step_update((epoch * steps_per_epoch) + batch_idx)


def train_epoch(model, train_loader, opts_lr_schedulers, epoch, steps_per_epoch, logger):
    model.train()

    loss_meter_sup_con = AverageMeter()
    loss_meter_cls = AverageMeter()
    loss_total = AverageMeter()
    grad_norm_meter_supCon = AverageMeter()
    grad_norm_meter_cls = AverageMeter()

    for idx, pack in enumerate(train_loader):

        _input = torch.cat([pack['image'][0], pack['image'][1]], dim=0).cuda()
        label = pack['label'].cuda()
        sup_con_logits, cls_logit = model(_input)
        sup_con_loss, cls_loss = criterion(sup_con_logits, cls_logit, label)
        take_step(model, sup_con_loss, cls_loss, idx, epoch, steps_per_epoch, opts_lr_schedulers, grad_norm_meter_supCon, grad_norm_meter_cls)

        loss_total.update((sup_con_loss.item() + cls_loss.item()))
        loss_meter_sup_con.update(sup_con_loss.item())
        loss_meter_cls.update(cls_loss.item())

        if idx % 10 == 0:
            _lr_sup_con = opts_lr_schedulers[0].param_groups[0]['lr']
            _lr_cls = opts_lr_schedulers[1].param_groups[0]['lr']
            mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

            logger.info(f'epoch[{epoch}][{idx}/{steps_per_epoch}]\t'
                             f'lr_sup_con={_lr_sup_con:.5f}\t'
                             f'lr_cls={_lr_cls:.5f}\t'
                             f'loss_sup_con={loss_meter_sup_con.avg:.5f}\t'
                             f'loss_cls={loss_meter_cls.avg:.5f}\t'
                             f'grad_norm_cls={grad_norm_meter_cls.avg:.5f}\t'
                             f'grad_norm_supCon={grad_norm_meter_supCon.avg:.5f}\t'
                             f'mem={mem:.2f}GB')


@torch.no_grad()
def validation(model, val_loader, logger):
    model.eval()

    _preds = []
    _labels = []
    _embeds = []

    for batch_idx, a_batch in enumerate(val_loader):
        images = a_batch['image'].cuda()
        labels = a_batch['label']
        sup_con_logits, cls_logits = model(images, mode='val')

        cls_preds = cls_logits.argmax(dim=1)
        _preds.extend(cls_preds.detach().cpu().numpy())
        _labels.extend(labels.numpy())
        _embeds.extend(sup_con_logits.detach().cpu().numpy())

    _labels = np.array(_labels)
    _preds = np.array(_preds)








