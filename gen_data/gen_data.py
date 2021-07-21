import os
import shutil
import torchvision.datasets as mnist_data
import wget


class DatasetParams:
    def __init__(self, dataset_config):
        self.mnist_root = dataset_config['mnist_root']
        self.mnist_link = dataset_config['mnist_link']
        self.version = dataset_config['version']
        self.transform = dataset_config['transform']


def make_database(dataset_param):
    """
    If there is no database in this project,
    this automatically generates the data.
    """
    mnist_link = dataset_param.mnist_link
    dataset_path = os.path.join("./", dataset_param.mnist_root)
    mnist_zip_file = "./mnist.zip"
    mnist_unzip_folder = "./mnist"

    if os.path.isdir(dataset_path) and \
        len(os.listdir(dataset_path)) == 0:
        shutil.rmtree(mnist_unzip_folder)
    if not os.path.isdir(mnist_unzip_folder):
        if not os.path.isfile(mnist_zip_file):
            print("Start downloading dataset...")
            wget.download(mnist_link)
            print("Finished downloading")
        print("start unzipping mnist.zip...")
        os.system("unzip mnist.zip -d ./mnist")
        print("finished unzipping")

    print("Start copying data files...")
    shutil.copytree(mnist_unzip_folder, dataset_path)
    print("Finished copying data files")


class DatasetGenerator:
    def __init__(self, dataset_config, save):
        self.dataset_param = DatasetParams(dataset_config)
        self.dataset_path = os.path.join("./", self.dataset_param.mnist_root)

    def gen_dataset(self):
        mnist = mnist_data.MNIST(root=self.dataset_path, train=True, download=True,
                                 transform=self.dataset_param.transform)
        return mnist


