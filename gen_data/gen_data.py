import os
import torchvision.datasets as mnist_data
import torchvision.transforms as transforms


class DatasetParams:
    def __init__(self, dataset_config):
        self.mnist_root = dataset_config['mnist_root']
        self.mnist_link = dataset_config['mnist_link']
        self.version = dataset_config['version']
        self.transform = dataset_config['transform']


class Dataset:
    def __init__(self, train_data, val_data):
        self.train_data = train_data
        self.val_data = val_data


class DatasetGenerator:
    def __init__(self, dataset_config, save=True):
        self.dataset_param = DatasetParams(dataset_config)
        self.dataset_path = os.path.join("./", self.dataset_param.mnist_root)
        self.save = save

    def gen_dataset(self):
        train_data = mnist_data.MNIST(root=self.dataset_path,
                                      train=True,
                                      download=self.save,
                                      transform=transforms.ToTensor())
        val_data = mnist_data.MNIST(root=self.dataset_path,
                                    train=False,
                                    download=self.save,
                                    transform=transforms.ToTensor())
        dataset = Dataset(train_data=train_data,
                          val_data=val_data)
        return dataset


