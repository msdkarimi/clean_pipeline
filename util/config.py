from ml_collections import ConfigDict
import os

EPOCHS = 200

def get_backbone_config():
    config = ConfigDict()
    config.name = 'focalnets'
    config.focal = ConfigDict()
    config.focal.embed_dim = 96
    config.focal.depths = [2, 2, 18, 2]
    config.focal.focal_levels = [3, 3, 3, 3]
    config.focal.focal_windows = [3, 3, 3, 3]
    config.focal.drop_path_rate = 0.5
    config.focal.fine_tune_modulation_block_adopter = False # if this is true, it freezes the vision backbone and just the adopter layer is trainable

    # config.focal.num_classes = 10
    return config

def get_opt_config(task='sup_con'):
    lr = 1e-3 # lr for backbone + sup_con head
    config = ConfigDict()
    if task == 'sup_con':
        config.name = 'SGD'
        config.base_lr = lr
        config.weight_decay = 1e-4
        config.momentum = 0.9
        return config

    elif task == 'cls':
        config.name = 'SGD'
        config.base_lr = lr * 10
        config.weight_decay = 1e-4
        config.momentum = 0.9
        return config

    else:
        raise ValueError(f'Unknown optimizer: {task}')

def get_lr_scheduler_config(method='linear'):
    config = ConfigDict()
    if method == 'linear':
        config.name = 'linear'
        config.decay_epochs = 0
        config.warmup_epochs = 2
        config.warmup_lr = 1e-5
        config.epochs = EPOCHS
        return config
    else:
        raise ValueError(f'Unknown method : {method} for lr_scheduler')
