import torch
from util.optimizer import build_optimizer
from util.lr_scheduler import build_scheduler
from util.config import get_opt_config, get_lr_scheduler_config
from util.loss import SupConLoss
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import matplotlib
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

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

def criterion(sup_con_logits, cls_logits, labels, cls_weights=[0.01458, 0.99578, 0.99330, 0.99632]):
    sup_con = SupConLoss()
    sup_con_loss = sup_con(sup_con_logits, labels)
    cls_loss = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(cls_weights)).cuda().float())(cls_logits, labels)
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
    # model.backbone.freezer()

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
def validation(model, val_loader, epoch, logger):
    model.eval()

    _preds = []
    _labels = []
    _embeds = []

    for batch_idx, a_batch in enumerate(val_loader):
        images = a_batch['image'].cuda()
        labels = a_batch['label']
        sup_con_logits, cls_logits = model(images, phase='val')
        cls_preds = cls_logits.argmax(dim=1)
        _preds.extend(cls_preds.detach().cpu().numpy())
        _labels.extend(labels.numpy())
        _embeds.extend(sup_con_logits.detach().cpu().numpy())

    _labels = np.array(_labels)
    _preds = np.array(_preds)

    cm = confusion_matrix(_labels, _preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    class_names = [f'Class_{lbl}' for lbl in np.unique(_labels).tolist()]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format=".2f")
    plt.title("Confusion Matrix")
    # plt.show()
    plt.savefig(f'plots/cm/e_{epoch}.png', format="png", dpi=300)

    precision = precision_recall_fscore_support(
            _labels , _preds, average='weighted', labels= np.unique(_labels)
        )

    precision, recall, f_score, true_sum = precision

    all_embeddings = _embeds
    all_labels = _labels

    scaler = StandardScaler()
    embeddings = scaler.fit_transform(all_embeddings)
    pca = PCA(n_components=10)
    pca_result = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(pca_result)

    sil_score = silhouette_score(embeddings, all_labels)
    distances = pairwise_distances(embeddings)

    # Initialize variables for inter-class and intra-class distances
    intra_class_distances = []
    inter_class_distances = []

    # Compute intra-class and inter-class distances
    for i in range(len(all_labels)):
        for j in range(i + 1, len(all_labels)):
            if all_labels[i] == all_labels[j]:
                intra_class_distances.append(distances[i, j])
            else:
                inter_class_distances.append(distances[i, j])

    # Calculate the mean intra-class and inter-class distances
    mean_intra_class_distance = np.mean(intra_class_distances)
    mean_inter_class_distance = np.mean(inter_class_distances)

    # Calculate the Inter/Intra Distance Ratio
    distance_ratio = mean_inter_class_distance / mean_intra_class_distance

    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=all_labels.tolist(), cmap='viridis', s=10)
    plt.colorbar()
    plt.xlabel(f'dimension 1')
    plt.ylabel(f'dimension 2')
    plt.title(f'Silhouette Score: {sil_score:.4f}, Inter/Intra-class Distance Ratio: {distance_ratio:.4f}')
    plt.savefig(f'plots/embed/e_{epoch}.png', format="png", dpi=300)  # You can change the filename and format if needed
    # plt.show()
    plt.close()

    logger.info(
        f'precision={precision}\t'
        f'recall={recall}\t'
        f'f_score={f_score}\t'
        f'sil_score={sil_score}\t'
        f'distance_ratio={distance_ratio:.4f}\t'
        # f'val_loss={(_cls_loss_tot / len(self.val_dataloader)):.4f}\t'
    )

    return f_score

def from_pretrained(model, load_dir, logger):
    state_dict = torch.load(load_dir)['model']
    state_dict = align_and_update_state_dicts(model.state_dict(), state_dict, logger)
    model.load_state_dict(state_dict, strict=False)
    return model

def align_and_update_state_dicts(model_state_dict, ckpt_state_dict, logger):
    model_keys = sorted(model_state_dict.keys())
    ckpt_keys = sorted(ckpt_state_dict.keys())
    result_dicts = {}
    matched_log = []
    unmatched_log = []
    unloaded_log = []
    for model_key in model_keys:
        model_weight = model_state_dict[model_key]
        if model_key in ckpt_keys:
            ckpt_weight = ckpt_state_dict[model_key]
            if model_weight.shape == ckpt_weight.shape:
                result_dicts[model_key] = ckpt_weight
                ckpt_keys.pop(ckpt_keys.index(model_key))
                matched_log.append("Loaded {}, Model Shape: {} <-> Ckpt Shape: {}".format(model_key, model_weight.shape,
                                                                                          ckpt_weight.shape))
            else:
                unmatched_log.append(
                    "*UNMATCHED* {}, Model Shape: {} <-> Ckpt Shape: {}".format(model_key, model_weight.shape,
                                                                                ckpt_weight.shape))
        else:
            unloaded_log.append("*UNLOADED* {}, Model Shape: {}".format(model_key, model_weight.shape))

    for info in matched_log:
        logger.info(info)
    for info in unloaded_log:
        logger.warning(info)
    for key in ckpt_keys:
        logger.warning("$UNUSED$ {}, Ckpt Shape: {}".format(key, ckpt_state_dict[key].shape))
    for info in unmatched_log:
        logger.warning(info)
    return result_dicts