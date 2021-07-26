import os
import shutil

import numpy as np
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torchvision.transforms as TT

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


def show_data(data, label):
    # data shape : torch.Size([Batch_size, 1, 28, 28])
    # label shape : torch.Size([4])
    num_row = 2
    num_col = 4
    fig, axes = plt.subplots(num_row, num_col)
    for i in range(8):
        ax = axes[i // num_col, i % num_col]
        ax.imshow(data[i, 0].cpu(), cmap='gray')
        ax.set_title(f'label : {label[i].cpu().item()}')
    plt.tight_layout()
    plt.show()


class Trainer:
    def __init__(self, config_list, dataset=None, save_model=False):
        train_params = TrainParams(config_list.train_config)
        self.data_root = 'dataset/mnist/MNIST/'
        self.num_epochs = train_params.num_epochs
        self.learning_rate = train_params.learning_rate
        self.batch_size = train_params.batch_size
        self.bta1 = train_params.bta1
        self.bta2 = train_params.bta2
        self.epsln = train_params.epsln
        self.save_model = save_model
        self.config_list = config_list

        self.train_ver = config_list.train_config['version']
        self.dataset_ver = config_list.dataset_config['version']
        self.bit_ver = config_list.bit_config['version']
        if dataset is None:
            dataset_gen = DatasetGenerator(dataset_config=self.config_list.dataset_config,
                                           save=True)
            self.dataset = dataset_gen.gen_dataset()
        else:
            self.dataset = dataset
        self.save = save_model
        self.toPIL = TT.ToPILImage()

    def create_loaders(self):
        train_loader = DataLoader(self.dataset.train_data,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=4)
        val_loader = DataLoader(self.dataset.val_data,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=4)

        return train_loader, val_loader

    def fit(self):
        # create directory and tensorboard Summary Writer for version
        version_all = '.'.join([self.dataset_ver,
                                self.train_ver,
                                self.bit_ver])
        writer_path = 'runs/%s' % version_all
        if os.path.isdir(writer_path):
            print('train version already exists. removing content.')
            shutil.rmtree(writer_path)
        writer = SummaryWriter(writer_path)

        model = QuadbitMnistModel(n_pixel=28)

        train_loader, val_loader = self.create_loaders()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            print("=====================================\n"
                  "GPU not available.\n"
                  "To stop current process, press ctrl+c.\n"
                  "=====================================")
        model.to(device)

        train_loss_iter = np.zeros(self.num_epochs, dtype=float)
        valid_loss_iter = np.zeros(self.num_epochs, dtype=float)
        train_accuracy_iter = np.zeros(self.num_epochs, dtype=float)
        valid_accuracy_iter = np.zeros(self.num_epochs, dtype=float)

        criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.learning_rate,
                                     betas=(self.bta1, self.bta2),
                                     eps=self.epsln)

        for epoch in range(self.num_epochs):
            # Training process beginning
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
                correct_cnt += (thresholding(prediction) ==
                                target.data).sum().item()

            accuracy = correct_cnt * 1.0 / total_cnt
            train_loss_iter[epoch] = total_loss / total_cnt
            train_accuracy_iter[epoch] = accuracy

            # Validation process beginning
            total_loss, total_cnt, correct_cnt = 0.0, 0.0, 0.0

            writer.add_scalar('Train_Loss', total_loss, epoch)
            writer.add_scalar('Train Accuracy', accuracy, epoch)

            for batch_idx, (x, target) in enumerate(val_loader):
                with torch.no_grad():
                    model.eval()
                    if torch.cuda.is_available():
                        x, target = x.cuda(), target.cuda()

                    prediction = model(x)
                    loss = criterion(prediction, target)

                    total_loss += loss.item()
                    total_cnt += x.data.size(0)
                    correct_cnt += (thresholding(prediction) == target.data).sum().item()

            accuracy = correct_cnt * 1.0 / total_cnt
            valid_loss_iter[epoch] = total_loss / total_cnt
            valid_accuracy_iter[epoch] = accuracy

            writer.add_scalar('Valid_Loss', total_loss, epoch)
            writer.add_scalar('Valid Accuracy', accuracy, epoch)

            if epoch % 10 == 0:
                print(f"[{epoch}/{self.num_epochs}]\n"
                      f"Train Loss : {train_loss_iter[epoch]:.4f} Train Acc : {train_accuracy_iter[epoch]:.2f} \n"
                      f"Valid Loss : {valid_loss_iter[epoch]:.4f} Valid Acc : {valid_accuracy_iter[epoch]:.2f}")

        writer.close()
        if self.save_model:
            save_root = f'models/{version_all}'
            os.makedirs(save_root)
            torch.save(model.state_dict(),
                       os.path.join(save_root, 'model.ckpt'))