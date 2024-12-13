import numpy as np
import pandas as pd
import torchvision
import omegaconf
from matplotlib import pyplot as plt
import wandb


def logging_images(wandb_run, batch, targets, mapping, num=16, per_class=False, is_train=True):
    if per_class:
        data_classes = np.unique(targets)
        for data_cls in data_classes:
            sub_batch = batch[targets == data_cls]
            title = f'{("Train" if is_train else "Val")} pics: class {mapping[int(data_cls)]}'
            wandb.log({f"{title}": [wandb.Image(sub_batch[:num])]})


def logging_data_stat(wandb_run, datasets, data_lists, is_train=True, targets_col='class'):
    group_cols = list(targets_col) if isinstance(targets_col, omegaconf.listconfig.ListConfig) else targets_col
    stat_dfs = [myset.take_df().groupby(group_cols).count().T[:1] for myset in datasets]
    full_stats = pd.concat(stat_dfs)
    # list_names = [name.split('/')[0] for name in data_lists]
    list_names = [name.replace('/', '_').replace('.txt', '') for name in data_lists]

    full_stats.index = list_names
    full_stats['Total'] = full_stats.sum(axis=1).astype(int)
    full_stats['type'] = [myset.data_type for myset in datasets]
    title = 'Datasets statistics'
    full_stats.fillna(0, inplace=True)
    wandb.log({f"{title}": [wandb.Table(full_stats.to_dict(orient='records'))]})


def logging_lr(wandb_run, value_to_log, step):
    wandb.run.log({"Learning Rate": value_to_log}, step=step)


def logging(wandb_run, metrics, value_to_log, step, is_train=True, per_epoch=False):
    if 'accuracy' in metrics:
        title = 'Accuracy'
    elif 'loss' in metrics:
        title = 'Running Loss'
    elif 'cm' in metrics:
        title = 'Conf Matrix'
    elif 'norm cm' in metrics:
        title = 'Normalized Conf Matrix'
    else:
        title = ''

    if is_train:
        title = ' '.join([title, 'over Epochs']) if per_epoch else ' '.join([title, 'over Steps'])
    else:
        title = ' '.join(['Train', title]) if per_epoch else ' '.join(['Val', title])

    if 'cm' in metrics:
        wandb.run.log({title: wandb.plot.confusion_matrix(value_to_log, xaxis='predictions', yaxis='labels', yaxis_reversed=True)}, step=step)
    else:
        wandb.run.log({title: value_to_log}, step=step)