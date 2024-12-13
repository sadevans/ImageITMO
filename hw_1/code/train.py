import os
import sys
import warnings
from datetime import datetime as dtm

import hydra
import numpy as np
import torch
import utils
from init_funcs import gather_and_init
from rich.console import Console
from torch import distributed as dist
from trainer import Trainer

warnings.filterwarnings('ignore')



@hydra.main(version_base=None)
def main(cfg):
    console = Console(record=True)
    model, train_options, wandb_run = gather_and_init(cfg)
    trainer = Trainer(model, cfg, train_options, wandb_run, console)
    best_loss = np.inf
    start_train = dtm.now()

    if train_options['rank'] == 0:
        utils.print_config(cfg, console)
        #utils.print_model(cfg, model, console)
        console.print('We are ready to start training! :rocket:', style='bold red')

    # main loop
    for epoch in range(1, cfg.train.n_epoch + 1):
        if train_options['rank'] == 0:
            console.print(f'[bold yellow]Epoch: [/]{epoch} / {cfg.train.n_epoch}')

        trainer.run_epoch(epoch)
        with torch.no_grad():
            val_loss = trainer.eval_epoch(epoch)

        if train_options['rank'] == 0:
            if val_loss <= best_loss:
                utils.save_checkpoint(model, train_options, epoch)
                best_loss = val_loss
            elif (epoch % 10 == 0) or (epoch == cfg.train.n_epoch):
                utils.save_checkpoint(model, train_options, epoch)

    delta_train = dtm.now() - start_train
    if train_options['rank'] == 0:
        console.print(f'Total elapsed training time: {utils.format_timedelta(delta_train)}', style='bold')
        console.print('Congrats, the whole experiment is finished! :tada:', style='bold red')
        console.save_html(os.path.join(cfg.logdir, ''.join(['logs_', cfg.exp_name, '.html'])))
        wandb_run.finish()

    dist.destroy_process_group()





if __name__ == '__main__':
    main()
