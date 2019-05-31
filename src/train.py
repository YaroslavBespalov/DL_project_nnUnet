import argparse
from pathlib import Path

import cv2
import torch

from dataset import TaskDataFactory
from youtrain.callbacks import ModelSaver, TensorBoard, Callbacks, Logger
from youtrain.factory import Factory
from youtrain.runner import Runner
from youtrain.utils import set_global_seeds, get_config, get_last_save

import warnings
warnings.filterwarnings('ignore')

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--paths', type=str, default=None)
    return parser.parse_args()


def create_callbacks(name, dumps, name_save, monitor_metric):
    log_dir = Path(dumps['path']) / dumps['logs'] / name
    save_dir = Path(dumps['path']) / dumps['weights'] / name
    callbacks = Callbacks([
        Logger(log_dir),
        ModelSaver(
            checkpoint=True,
            metric_name=monitor_metric,
            save_dir=save_dir,
            save_every=1,
            save_name=name_save,
            best_only=True,
            threshold=0.7),
#        TensorBoard(log_dir)
    ])
    return callbacks


def main():
    args = parse_args()
    # set_global_seeds(42)
    config = get_config(args.config)

    print(config)
    print()

    config['train_params']['name'] = f'{config["train_params"]["name"]}/{args.fold}'
    paths = get_config(args.paths)

    print(paths)
    print()

    if config['train_params']['new_save']:
        paths["dumps"]['name_save'] = f'{paths["dumps"]["name_save"]}_{get_last_save(Path(paths["dumps"]["path"]) / paths["dumps"]["weights"] / config["train_params"]["name"]) + 1}'
    else:
        paths["dumps"]['name_save'] = paths['name_save']

    config['train_params']['name_save'] = paths["dumps"]['name_save']
    config['train_params']['save_dir'] = Path(paths['dumps']['path']) / paths['dumps']['weights'] / config['train_params']['name']
    factory = Factory(config['train_params'])

    data_factory = TaskDataFactory(config['data_params'], paths['data'], fold=args.fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    callbacks = create_callbacks(config['train_params']['name'], paths['dumps'], paths["dumps"]["name_save"], config['train_params']['metrics'][-1])

    trainer = Runner(
        stages=config['stages'],
        factory=factory,
        callbacks=callbacks,
        device=device,
        fold=args.fold
    )

    trainer.fit(data_factory)


if __name__ == '__main__':
    main()
