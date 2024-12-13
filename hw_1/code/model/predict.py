import argparse
import os
import sys

import pandas as pd
import torch
from data import get_test_loader
from omegaconf import OmegaConf
from rich.console import Console
from tqdm import tqdm
from utils import load_checkpoint, load_model

console = Console()


def inference(model, config, data_list, data_type, root):
    console.print(f'Prediction for [blue1]{data_list}[/] started')
    test_loader = get_test_loader(config, data_list, data_type, root, is_inference=True)
    test_len = len(test_loader)

    with torch.no_grad():
        for ind, (images, impaths) in tqdm(enumerate(test_loader, 0), total=test_len):
            outputs = model(images.cuda())

            if config.model.num_classes == 1:
                outputs = torch.sigmoid(outputs)
                outputs = outputs.squeeze().detach().cpu().numpy()
                df = pd.DataFrame(zip(impaths, outputs))
            else:
                outputs = torch.softmax(outputs, dim=1)
                scores, preds = torch.max(outputs, dim=1)
                scores = scores.detach().cpu().tolist()
                preds = preds.detach().cpu().tolist()
                df = pd.DataFrame(zip(impaths, preds, scores), columns=['image_name', 'pred', 'score'])

            if ind == 0:
                pred_result = df
            else:
                pred_result = pd.concat((pred_result, df))

        console.print(f'Prediction for [blue1]{len(pred_result)}[/] items is done!')
    return pred_result


def main(args):
    cfg = OmegaConf.load(args.cfg)
    # getting model and checkpoint
    model = load_model(cfg)
    model.load_state_dict(load_checkpoint(cfg))
    model.eval().cuda()

    pred_result = inference(model, cfg, args.file_list, args.data_type, args.root)

    # Saving results
    test_list_name = args.file_list.split('/')[-1]
    save_dir = args.save_dir

    save_path = os.path.join(save_dir, f'{cfg.exp_name}_{test_list_name}.csv')
    pred_result.to_csv(save_path, index=False)

    console.print(f'Get predictions here: [blue1]{save_path}[/]')
    console.print("Congrats! You've done prediction for the dataset. :tada:", style='red bold')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='Config file to use')
    parser.add_argument('--file_list', type=str, help='Filelist with imnames for inference')
    parser.add_argument('--root', type=str, help='Image prefix')
    parser.add_argument('--data_type', type=str, default='full_frame', help='options: full_frame/from_bbox')
    parser.add_argument('--save_dir', type=str, help='Filelist with imnames for inference')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
