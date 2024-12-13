import os

import hydra
import pandas as pd
import torch
import utils
from init_funcs import gather_and_init
from data import get_test_loader
from rich.console import Console
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score
from tqdm import tqdm


def calculate_pr_rec(pred_result, exp, base, console, gt_column='class', pred_column='pred'):
    y_true = pred_result[gt_column]
    y_pred = pred_result[pred_column]
    acc = accuracy_score(y_true, y_pred)
    av_rec = balanced_accuracy_score(y_true, y_pred)
    av_pr_macro = precision_score(y_true, y_pred, average='macro')
    av_f1_macro = f1_score(y_true, y_pred, average='macro')
    console.print('Accuracy for {0} on {1} is {2:.4f}'.format(exp, base, acc))
    console.print('Average recall for {0} on {1} is {2:.4f}'.format(exp, base, av_rec))
    console.print('Average precision (macro) for {0} on {1} is {2:.4f}'.format(exp, base, av_pr_macro))
    console.print('Average f1 for {0} on {1} is {2:.4f}'.format(exp, base, av_f1_macro))


def evaluation(model, config, data_list, data_type):  # noqa: WPS210
    test_loader = get_test_loader(config, data_list, data_type)
    test_len = len(test_loader)
    with torch.no_grad():
        for ind, (images, labels, impaths, bboxes) in tqdm(enumerate(test_loader, 0), total=test_len):
            outputs = model(images.cuda())

            if config.test.activation == 'sigmoid':
                outputs = torch.sigmoid(outputs)
            elif config.test.activation == 'softmax':
                outputs = torch.softmax(outputs, dim=1)

            scores, preds = torch.max(outputs, 1)
            scores = scores.data.tolist()
            preds = preds.data.cpu().tolist()
            labels = labels.data.tolist()
            if torch.all(bboxes == 0):
                df = pd.DataFrame(
                    zip(impaths, labels, preds, scores),
                    columns=['image_name', 'class', 'pred', 'score'],
                )

            else:
                x1, y1, x2, y2 = zip(*[bbox_list[:4] for bbox_list in bboxes.numpy().tolist()])
                df = pd.DataFrame(
                    zip(impaths, x1, y1, x2, y2, labels, preds, scores),
                    columns=['image_name', 'x1', 'y1', 'x2', 'y2', 'class', 'pred', 'score'],
                )

            if ind == 0:
                pred_result = df
            else:
                pred_result = pd.concat((pred_result, df))

        return pred_result


@hydra.main(version_base=None)
def main(cfg):
    console = Console(record=True)
    model = utils.load_model(cfg)
    model.cuda()
    model.eval()
    save_folder = os.path.join(cfg.outdir, cfg.exp_name)
    utils.print_config(cfg, console)

    data_lists = cfg.dataset.test_lists
    if hasattr(cfg.dataset, 'test_data_types'):
        data_types = cfg.dataset.test_data_types
        assert len(data_types) == len(data_lists), 'we need type labels for all sets'
    else:
        data_types = ['full_frame' for _ in range(len(data_lists))]

    for (data_list, data_type) in zip(data_lists, data_types):
        console.print('______________________________________________________', style='green bold')
        console.print(f'Evaluation for [dodger_blue2]{data_list}[/] started')
        pred_result = evaluation(model, cfg, data_list, data_type)
        # Saving results
        test_list_name = data_list.split('.')[0].replace('/', '_')
        save_path = os.path.join(save_folder, f'{cfg.exp_name}_{test_list_name}.csv')
        pred_result.to_csv(save_path, index=False)

        console.print('Results for main head:', style='red green')
        calculate_pr_rec(pred_result, cfg.exp_name, test_list_name, console, gt_column='class', pred_column='pred')
        if 'pred1' in pred_result.columns:
            console.print('Results for additional head:', style='red green')
            calculate_pr_rec(pred_result, cfg.exp_name, test_list_name, console, gt_column='class1', pred_column='pred1')

        console.print(f'Get predictions here: [dodger_blue2]{save_path}[/]')
        console.print('Congrats! You have done evaluation for the dataset. :tada:', style='red bold')

    console.print('______________________________________________________', style='green bold')
    console.save_html(os.path.join(cfg.logdir, ''.join(['eval_', cfg.exp_name, '.html'])))


if __name__ == '__main__':
    main()
