import argparse
import json
import os

from dataclasses import dataclass

BIT_CONFIG_FILE = 'bit_config.json'
DATASET_CONFIG_FILE = 'dataset_config.json'
TRAIN_CONFIG_FILE = 'train_config.json'


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


@dataclass
class Configs:
    def __init__(self, config_root, versions):
        self.bit_config = load_config(
            os.path.join(config_root, BIT_CONFIG_FILE), versions[0]
        )
        try:
            self.dataset_config = load_config(
                os.path.join(config_root, DATASET_CONFIG_FILE), versions[2]
            )
            self.train_config = load_config(
                os.path.join(config_root, TRAIN_CONFIG_FILE), versions[1]
            )
        except IOError:
            pass
        self.version_all = '.'.join(versions[:-1])


def load_config(config_path, version):
    with open(config_path, 'rb') as f:
        config_total = json.load(f)
    result = config_total[version]
    if config_total.get('roots') is not None:
        result.update(config_total['roots'])
    if config_total.get('links') is not None:
        result.update(config_total['links'])
    result['version'] = version
    return result


def create_data(config_list, save: bool = True):
    from gen_data import DatasetGenerator
    gen_data = DatasetGenerator(config_list.dataset_config, save=save)
    return gen_data.gen_dataset()


def train(config_list, dataset=None, save_model=False, quantize=False):
    from train import Trainer
    trainer = Trainer(config_list,
                      dataset=dataset,
                      save_model=save_model)
    trainer.fit(quantize=quantize)


def parse_ver(version_raw):
    """
    Parses period split values of different versions and returns dataclass
    specifying version types
    Args:
        version_raw: str
            period split values, e.g. 0.0.0.0
            bit_config_version . train_config_version . data . extra bit
    Returns: dataclass versions
        list of  str versions for each modes
    """
    version_list = version_raw.split('.')
    if len(version_list) > 4:
        raise ValueError(
            'Invalid version format, upto 4 versions required: '
            'bit.train.data.sth'
        )
    return version_list


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--mode', type=str,
                   choices=['data', 'bit', 'train', 'all'],
                   default='all')
    p.add_argument('-v', '--version', type=str, default='0.0.0.0')
    p.add_argument('--config_root', type=str, default='./assets')
    p.add_argument('-s', '--save', type=bool, default=False)
    p.add_argument('--save_model', type=bool, default=False)
    p.add_argument('--quantize_train', type=bool, default=False)
    args = p.parse_args()

    parsed_ver = parse_ver(args.version)
    config_list = Configs(config_root=args.config_root, versions=parsed_ver)
    if args.mode == 'data':
        create_data(config_list)
    elif args.mode == 'train':
        train(config_list)
    else:
        data = create_data(config_list, save=False)
        train(config_list, dataset=data,
              save_model=args.save_model,
              quantize=args.quantize_traian)


if __name__ == '__main__':
    main()