import os
import data_utils
import torch
from hydra.utils import instantiate
from logger import logging_data_stat
import torch.distributed as dist
import numpy as np


def get_train_loader(config, task=None, rank=0):
    transform = data_utils.get_transform(config, is_train=True)
    train_lists = config.dataset.train_lists

    if hasattr(config.dataset, 'train_data_types'):
        data_types = config.dataset.train_data_types
        assert len(data_types) == len(train_lists), 'we need type labels for all sets'
    else:
        data_types = ['full_frame' for i in range(len(train_lists))]
    
    datasets = data_utils.get_datasets(
        config.dataset.dataset_class,
        train_lists,
        config.dataset.root,
        config.dataset.prefix_lists,
        transform,
        data_types,
        config.model.num_classes,
        config.dataset.class_weights if hasattr(config.dataset, 'class_weights') else None,
        targets_column=config.dataset.targets_column,
    )

    if rank == 0:
        logging_data_stat(
            task.get_logger(),
            datasets,
            train_lists,
            targets_col=config.dataset.targets_column,
        )

    train_set = torch.utils.data.ConcatDataset(datasets)

    data_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    return data_loader  # noqa: WPS331


def get_val_loader(config, task=None, rank=0):
    transform = data_utils.get_transform(config, is_train=False)
    val_lists = config.dataset.val_lists

    if hasattr(config.dataset, 'val_data_types'):
        data_types = config.dataset.val_data_types
        assert len(data_types) == len(val_lists), 'we need type labels for all sets'
    else:
        data_types = ['full_frame' for i in range(len(val_lists))]

    datasets = data_utils.get_datasets(
        config.dataset.dataset_class,
        val_lists,
        config.dataset.root,
        config.dataset.prefix_lists,
        transform,
        data_types,
        config.model.num_classes,
        config.dataset.class_weights if hasattr(config.dataset, 'class_weights') else None,
        targets_column=config.dataset.targets_column,
    )

    if rank == 0:
        logging_data_stat(
            task.get_logger(),
            datasets,
            val_lists,
            is_train=False,
            targets_col=config.dataset.targets_column,
        )

    val_set = torch.utils.data.ConcatDataset(datasets)

    data_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return data_loader  # noqa: WPS331


def get_test_loader(config, data_list, data_type, root='', is_inference=False):
    transform = data_utils.get_transform(config, is_train=False)
    if is_inference:
        # test_dataset = instantiate(
        #     config.dataset.dataset_class,
        #     root=root,
        #     annotation_file=data_list,
        #     data_type=data_type,
        #     transform=transform,
        #     is_inference=is_inference,
        # )
        test_dataset = instantiate(
            config.dataset.dataset_class,
            config.dataset.root,
            data_list,
            transform,
            data_type,
            config.model.num_classes,
            targets_column=config.dataset.targets_column,
            class_weights=config.dataset.class_weights if hasattr(config.dataset, 'class_weights') else None,
        )
    else:
        data_list = os.path.join(config.dataset.prefix_lists, data_list)
        # test_dataset = instantiate(
        #     config.dataset.dataset_class,
        #     root=config.dataset.root,
        #     annotation_file=data_list,
        #     data_type=data_type,
        #     transform=transform,
        #     targets_column=config.dataset.targets_column,
        # )
        test_dataset = instantiate(
            config.dataset.dataset_class,
            config.dataset.root,
            data_list,
            transform,
            data_type,
            config.model.num_classes,
            targets_column=config.dataset.targets_column,
            class_weights=config.dataset.class_weights if hasattr(config.dataset, 'class_weights') else None,
        )

    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        drop_last=False,
        pin_memory=True,
    )
    return data_loader  # noqa: WPS331


def get_dataloader(config, split_type, task=None, rank=0):
    if split_type == 'train':
        data_loader = get_train_loader(config, task, rank)
    elif split_type == 'val':
        data_loader = get_val_loader(config, task, rank)
    else:
        raise KeyError('Subset unexpected type:', split_type)

    return data_loader
