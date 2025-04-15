from torch import optim as optim

def build_optimizer(config, models):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if not isinstance(models, list):
        models = [models]
    for model in models:
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        if hasattr(model, 'no_weight_decay_keywords'):
            skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(models, config.base_lr, skip, skip_keywords)

    opt_name = config.name.lower()
    optimizer = None
    if opt_name == 'sgd':
        optimizer = optim.SGD(parameters, momentum=float(config.momentum), nesterov=True,
                              lr=float(config.base_lr), weight_decay=float(config.weight_decay))
        return optimizer
    # elif opt_name == 'adamw':
    else:
        raise ValueError(f'not implemented {opt_name}')
        # optimizer = optim.AdamW(parameters, eps=float(config["TRAIN"]["OPTIMIZER"]["EPS"]), betas=(float(config["TRAIN"]["OPTIMIZER"]["BETAS_0"]), float(config["TRAIN"]["OPTIMIZER"]["BETAS_1"])),
        #                         lr=float(config["TRAIN"]["BASE_LR"]), weight_decay=float(config["TRAIN"]["WEIGHT_DECAY"]))

    # return optimizer


def set_weight_decay(models, base_lr, skip_list=(), skip_keywords=() ):
    has_decay = []
    no_decay = []
    cls_head = []
    for model in models:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                    check_keywords_in_name(name, skip_keywords):
                no_decay.append(param)
            else:
                # if "cls_head" not in name:
                has_decay.append(param)

            # if "cls_head" in name and not name.endswith(".bias"):
            #     cls_head.append(param)

    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}
            ]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin