import os
from itertools import product

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms


class Inferencer:
    def __init__(self, path_to_checkpoint, size=224):
        self.class_map = {
            0: 'none',
            1: 'fire',
            2: 'white_smoke',
            3: 'black_smoke',
        }
        self.model = self.load_model(path_to_checkpoint)
        self.size = size
        self.transform = self.create_transform()

    def create_transform(self):
        transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ])
        return transform

    def preprocessing(self, crop):
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = Image.fromarray(crop)
        tensor = self.transform(crop)
        return tensor

    def load_checkpoint(self, path_to_checkpoint):
        checkpoint = torch.load(os.path.join(path_to_checkpoint))['state_dict']
        new_state_dict = {}
        for key, weight in checkpoint.items():
            name = key.replace('module.', '')
            name = name.replace('_orig_mod.', '')
            new_state_dict[name] = weight
        return new_state_dict

    def load_model(self, path_to_checkpoint):
        model = models.resnet50(weights='DEFAULT')
        num_classes = len(self.class_map)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(self.load_checkpoint(path_to_checkpoint))
        model = model.eval()
        return model

    def slice_frame(self, frame, h_num_blocks, w_num_blocks):
        h, w, _ = frame.shape
        h_block = h // h_num_blocks
        w_block = w // w_num_blocks
        grid = product(
            range(0, w - w % w_block, w_block),
            range(0, h - h % h_block, h_block),
        )
        all_boxes = [(x, y, w_block, h_block) for (x, y) in grid]
        batch = [frame[y:y + h, x:x + w] for (x, y, w, h) in all_boxes]  # noqa: WPS221
        return all_boxes, batch

    def run_model(self, batch):
        outputs = torch.softmax(self.model(batch), dim=1)
        scores, preds = torch.max(outputs, 1)
        scores, preds = scores.tolist(), preds.tolist()
        names = [self.class_map[pred] for pred in preds]
        scores = [np.round(score, 4) for score in scores]
        return names, scores

    def prepare_batch(self, frame, h_num_blocks, w_num_blocks):
        rects, batch = self.slice_frame(frame, h_num_blocks, w_num_blocks)
        tensor_list = [self.preprocessing(crop) for crop in batch]
        tensor_batch = torch.stack(tensor_list)
        return rects, tensor_batch

    def run_inference(self, frame, h_num_blocks=1, w_num_blocks=1):
        rects, batch = self.prepare_batch(frame, h_num_blocks, w_num_blocks)
        names, scores = self.run_model(batch)
        res = [{'objectType': name, 'rect': rect, 'score': score} for (name, rect, score) in zip(names, rects, scores)]
        return res
