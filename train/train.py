import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import tqdm
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import QuadbitMnistModel
from gen_data import DatasetGenerator

@dataclass()
class TrainParams:
    def __init__(self, train_config, **_extras):
        super(TrainParams, self).__init__(**_extras)
        self.num_epochs = train_config['num_epochs']
        self.learning_rate = train_config['learning_rate']
        self.batch_size = train_config['batch_size']
        self.bta1 = train_config['bta1']
        self.bta2 = train_config['bta2']
        self.epsln = train_config['epsln']


def thresholding(prediction):
    _, pred_label = torch.max(prediction, 1)

    return pred_label


class Trainer:
    def __init__(self, config_list, dataset=None, save_model=False):
        train_params = TrainParams(config_list.train_config)
        feature_version = '.'.join([config_list.dataset_config['version'],
                                    config_list.bit_config['version'],
                                    config_list.train_config['version']])
        self.data_root = 'dataset/mnist/MNIST/raw'
        self.num_epochs = train_params.num_epochs
        self.learning_rate = train_params.learning_rate
        self.batch_size = train_params.batch_size
        self.bta1 = train_params.bta1
        self.bta2 = train_params.bta2
        self.epsln = train_params.epsln

        self.config_list = config_list

        self.train_ver = config_list.train_config['version']
        self.dataset_ver = config_list.dataset_config['version']
        self.bit_ver = config_list.bit_config['version']

        self.dataset = dataset
        self.save = save_model

    def create_loaders(self):
        if self.dataset is None:
            DatasetGen = DatasetGenerator(
                dataset_config=self.config_list.dataset_config,
                save=False)
            dataset = DatasetGen.gen_dataset()
        else:
            dataset = self.dataset

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset=dataset, lengths=[round(.8 * len(dataset)),
                                      round(.2 * len(dataset))], )

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=4)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=self.batch_size,
                                shuffle=True)
        return train_loader, val_loader

    def fit(self):
        version_all = '.'.join([self.dataset_ver,
                                self.train_ver,
                                self.bit_ver])
        writer_path = 'runs/%s' % version_all
        if os.path.isdir(writer_path):
            print('train version already exists. removing content.')
            shutil.rmtree(writer_path)
        writer = SummaryWriter(writer_path)

        train_loss_iter = np.zeros(self.num_epochs, dtype=float)
        valid_loss_iter = np.zeros(self.num_epochs, dtype=float)
        train_accuracy_iter = np.zeros(self.num_epochs, dtype=float)
        valid_accuracy_iter = np.zeros(self.num_epochs, dtype=float)

        train_loader, val_loader = self.create_loaders()
        dataloaders = {'train': train_loader, 'val': val_loader}

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            print('Warning: Cuda device not found. training with cpu')

        model = QuadbitMnistModel(n_pixel=28)
        criterion = nn.CrossEntropyLoss()

        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.learning_rate,
                                     betas=[self.bta1, self.bta2],
                                     eps=self.epsln)
        prog_bar = tqdm.tqdm(desc='training in progress',
                             total=self.num_epochs,
                             position=0,
                             leave=True)
        for epoch in range(self.num_epochs):
            total_loss, total_cnt, correct_cnt = 0.0, 0.0, 0.0

            for batch_idx, (x, target) in enumerate(train_loader):
                if torch.cuda.is_available():
                    x, target = x.cuda(), target.cuda()

                prediction = model(x)

                optimizer.zero_grad()
                loss = criterion(prediction, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_cnt += x.data.size(0)
                correct_cnt += (thresholding(prediction) == target.data).sum().item()

            accuracy = correct_cnt * 1.0 / total_cnt
            train_loss_iter[epoch] = total_loss / total_cnt
            train_accuracy_iter[epoch] = accuracy

            total_loss, total_cnt, correct_cnt = 0.0, 0.0, 0.0

            for batch_idx, (x, target) in enumerate(val_loader):
                with torch.no_grad():
                    if torch.cuda.is_available():
                        x, target = x.cuda(), target.cuda()

                    prediction = model(x)
                    loss = criterion(prediction, target)

                    total_loss += loss.item()
                    total_cnt += x.data.size(0)
                    correct_cnt += (thresholding(prediction) == target.data).sum().item()

            accuracy = correct_cnt * 1.0 / total_cnt
            train_loss_iter[epoch] = total_loss / total_cnt
            train_accuracy_iter[epoch] = accuracy

            if epoch % 10 == 0:
                print(f"[{epoch}/{self.num_epochs}] Train Loss : {train_loss_iter[epoch]:.4f} Train Acc : {train_accuracy_iter[epoch]:.2f} \
                Valid Loss : {valid_loss_iter[epoch]:.4f} Valid Acc : {valid_accuracy_iter[epoch]:.2f}")
            prog_bar.update()
        prog_bar.close()
        writer.close()
        if self.save:
            save_root = f'models/{version_all}'
            os.makedirs(save_root)
            torch.save(model.state_dict(),
                       os.path.join(save_root, 'model.ckpt'))
