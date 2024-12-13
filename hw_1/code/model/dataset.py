import os
import warnings
import torch
import cv2
import jpeg4py
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils import data
import numpy as np
from omegaconf.listconfig import ListConfig

warnings.filterwarnings('ignore')


def get_classes_weights(flists, num_classes, class_weights):
    print(flists[0])
    full_data = pd.read_csv(flists[0])
    full_data = full_data[['image_name', 'class']]
    for i, flist in enumerate(flists[1:]):
        full_data = full_data.append(pd.read_csv(flist)[['image_name', 'class']])

    full_data = full_data.dropna()
    full_data['class'] = pd.to_numeric(full_data['class']) 

    num_per_class = np.ones(num_classes)
    weights_per_item = np.zeros(num_classes, dtype=np.float64)

    num_per_class_df = full_data.groupby('class').count().reset_index()
    for cl in np.unique(full_data['class'].values):
        count = num_per_class_df[num_per_class_df['class'] == cl]['image_name'].values
        if len(count) > 0:
            num_per_class[int(cl)] = count[0]
        else:
            num_per_class[int(cl)] = 0

    if class_weights is None:
        weights = np.ones(len(full_data))

    elif isinstance(class_weights, str) and class_weights.lower() == 'balanced':
        for cl in np.unique(full_data['class'].values):
            weights_per_item[int(cl)] = 1. / num_per_class[int(cl)]
        weights = np.zeros(len(full_data))
        for i in range(len(full_data)):
            weights[i] = weights_per_item[int(full_data['class'].iloc[i])]

    elif isinstance(class_weights, ListConfig):
        if len(class_weights) != num_classes:
            raise ValueError(f"Num weights ({len(class_weights)}) doesn't equal to num classes ({num_classes})")
        weights_per_item = np.array(class_weights)
        weights = np.zeros(len(full_data))
        for i in range(len(full_data)):
            weights[i] = weights_per_item[int(full_data['class'].iloc[i])]
    else:
        raise ValueError("Invalid format for class_weights")

    return weights


def read_opencv(image_file):
    img = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise ValueError(f'Failed to read {image_file}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def read_image(image_file):
    if image_file.split('.')[-1] in {'jpg', 'jpeg'}:
        try:
            img = jpeg4py.JPEG(image_file).decode()
        except Exception as e:  # noqa: WPS424
            print(f'It is not jpg in fact -> {image_file}')
            img = read_opencv(image_file)
    else:
        img = read_opencv(image_file)
    return img


class MultiClassDataset(data.Dataset):
    def __init__(
        self,
        root,
        annotation_file,
        transform,
        data_type,
        num_classes,
        class_weights,
        is_inference=False,
        targets_column='class',
        rank=0,
    ):
        # super().__init__(root, annotation_file, transform, data_type, num_classes, class_weights, is_inference, targets_column=targets_column)
        
        self.root = root
        self.targets_column = targets_column
        self.df = pd.read_csv(annotation_file)
        self.df = self.df[~self.df[targets_column[0]].isna()]
        self.imlist = self.df.values.tolist()
        self.transform = transform
        self.is_inference = is_inference
        self.data_type = data_type
        if not self.is_inference:
            self.weights = get_classes_weights([annotation_file], num_classes, class_weights=class_weights)
        self.targets = list(self.df[self.targets_column[0]].unique())

        self.targets_column = targets_column
        for column in self.df.columns:  # noqa: WPS426
            if 'probs' in column:
                probs_column = self.df[column]
                probs_column = probs_column.apply(lambda x: x[1:-1].split(', '))
                probs_column = probs_column.apply(lambda x: np.array(list(map(float, x))))
                self.df[column] = probs_column
        self.imlist = self.df.values.tolist()

    def make_crop(self, img, coords):
        if coords is not None:
            if len(coords) == 4:
                x1, y1, x2, y2 = coords
            elif len(coords) == 5:
                x1, y1, x2, y2, _ = coords

            if x1 != np.nan and y1 != np.nan:
                x1, y1 = max(int(x1), 0), max(int(y1), 0)
            else:
                x1, y1 = 0, 0

            if x2 != np.nan and y2 != np.nan:
                x2 = min(int(x2), img.shape[1])
                y2 = min(int(y2), img.shape[0])
            else:
                x2 = img.shape[1]
                y2 = img.shape[0]

            return img[y1: y2, x1: x2]
        else:
            return img

    def __getitem__(self, index):
        data_item = self.df.iloc[index]

        image_name = data_item['image_name']

        full_imname = image_name
        if not os.path.exists(image_name):
            full_imname = os.path.join(self.root, image_name)

        if os.path.exists(full_imname):
            img = read_image(full_imname)
        else:
            raise FileNotFoundError(f'No such pic! Check the path {full_imname}!')

        bbox = None
        if self.data_type == 'from_bbox':
            if 'x_1' in data_item and not np.isnan(data_item['x_1']):
                bbox = [int(data_item['x_1']), int(data_item['y_1']), int(data_item['x_2']), int(data_item['y_2'])]
            elif 'x1' in data_item and not np.isnan(data_item['x1']):
                bbox = [int(data_item['x1']), int(data_item['y1']), int(data_item['x2']), int(data_item['y2'])]
            elif 'bbox' in data_item and data_item['bbox'] is not None:
                bbox = data_item['bbox']
            img = self.make_crop(img, bbox)
        try:
            img = Image.fromarray(img)
        except Exception:
            print(f'Problems with that img -> {image_name}')

        target = data_item.get(self.targets_column)
        if isinstance(target, pd.Series):
            target = target.to_list()

        if len(target) == 1:
            target = target[0]

        img, target = self.transform(img, target)

        if self.is_inference:
            return img, image_name

        if bbox is not None:
            bbox_tensor = torch.tensor(bbox, dtype=torch.int32)
        else:
            bbox_tensor = torch.tensor([0, 0, 0, 0], dtype=torch.int32)

        return img, target, image_name, bbox_tensor
