import os


import warnings
import torch
from hydra.utils import instantiate
from torchvision.transforms import v2

import augs

warnings.filterwarnings('ignore')


def get_datasets(
    dataset_class,
    data_lists,
    root,
    prefix_lists,
    transform,
    data_types,
    num_classes,
    class_weights,
    targets_column='class',
):
    datasets = []
    for (data_list, data_type) in zip(data_lists, data_types):
        dataset = instantiate(
            dataset_class,
            root,
            os.path.join(prefix_lists, data_list),
            transform,
            data_type,
            num_classes,
            targets_column=targets_column,
            class_weights=class_weights,
        )
        datasets.append(dataset)

    return datasets


class CustomCompose:
        def __init__(self, transforms):
            self.transforms = transforms
        
        def __call__(self, img, target=None):
            for t in self.transforms:
                if hasattr(t, 'p') and isinstance(t, augs.RandomHorizontalFlip):
                    img, target = t(img, target)
                else:
                    img = t(img)
            
            return img, target


def get_transform(config, is_train=False):
    if is_train and ('augmentation' in config):
        augs = []
        for aug_name, aug_item in config.augmentation.items():
            if aug_name == 'to_dtype':
                aug_object = instantiate(aug_item, dtype=torch.float32)
            else:
                aug_object = instantiate(aug_item, _convert_='all')
            augs.append(aug_object)

        return CustomCompose(augs)
    else:
        transform = CustomCompose([
            v2.PILToTensor(),
            v2.Resize((config.dataset.h, config.dataset.w)),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(mean=[0.5], std=[0.5]),
        ])

    return transform


def cal_el_weights(datasets, weights):
    eweights = []
    for ind, weight in enumerate(weights):
        prob = 1 / len(datasets[ind])
        cur_list = [weight * prob for elem in range(len(datasets[ind]))]
        eweights.append(cur_list)

    return [element for sublist in eweights for element in sublist]
