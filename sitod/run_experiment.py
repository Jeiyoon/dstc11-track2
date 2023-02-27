from setproctitle import *
setproctitle('k4ke')

import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"

import argparse
import logging
from allennlp.common import Params
from sitod.experiment import Experiment


def main(data_root_dir: str, experiment_root_dir: str, config: str) -> None:
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    params = Params.from_file(config)
    experiment = Experiment.from_params(params)
    experiment.run_experiment(data_root_dir=data_root_dir, experiment_root_dir=experiment_root_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir', type=str, required=True)
    parser.add_argument('--experiment_root_dir', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    main(data_root_dir=args.data_root_dir,
         experiment_root_dir=args.experiment_root_dir,
         config=args.config)
