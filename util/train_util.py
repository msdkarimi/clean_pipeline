import torch
from util.optimizer import build_optimizer
from util.lr_scheduler import build_scheduler
from util.config import get_opt_config, get_lr_scheduler_config

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


    lr_scheduler_1 = build_scheduler(get_lr_scheduler_config(), optimizer_sup_con, num_training_steps_per_epoch)
    lr_scheduler_2 = build_scheduler(get_lr_scheduler_config(), optimizer_cls, num_training_steps_per_epoch)

    return (
        optimizer_sup_con, optimizer_cls , lr_scheduler_1, lr_scheduler_2
    )

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()

    loss_meter = AverageMeter()
    grad_norm_meter = AverageMeter()
    param_norm_meter = AverageMeter()

    for idx, pack in enumerate(train_loader):

        _input = torch.cat([pack['image'][0], pack['image'][1]], dim=0)
        output = model(_input)
