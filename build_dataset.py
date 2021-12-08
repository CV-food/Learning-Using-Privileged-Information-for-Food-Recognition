import io

import numpy
import scipy.io as matio
import numpy as np
from PIL import Image

import torch.utils.data
from torchvision import transforms

'''
ingredient_{train/test/val}_feature.mat:
contains a matrix of (66071/11016/33154,353), indicating the ingredient representation of train/val/test data

indexVector_train(test).mat:
contains a matrix of (77087/33154,30)
(range from 1-309 to fit the requirement of Embedding layer that needs a zero vector on the top)
is the input to ingredient channel (to fit the embedding layer of gru encoder),
storing the index of words for individual food, with a maximum number of 30 (padding with zeros for entries without words).
'''


def default_loader(image_path):
    return Image.open(image_path).convert('RGB')


def load_ingredient(path) -> dict[str, type(numpy.zeros(353))]:
    d = {}
    with io.open(path + 'IngreLabel.txt', encoding='utf-8') as file:
        lines = file.read().split('\n')[:-1]
        for line in lines:
            l = line.split(' ')
            # print(len(l))
            s = l[0]
            t = l[1:]
            j = numpy.zeros(len(t))
            for i in range(len(t)):
                if t[i] == '1':
                    j[i] = 1
                elif t[i] == '-1':
                    j[i] = 0
                else:
                    assert (0)
            d[s] = j
    return d


ingredient_library = load_ingredient('./SplitAndIngreLabel/')


class dataset_stage1(torch.utils.data.Dataset):
    def __init__(self, dataset_indicator, image_path=None, data_path=None, transform=None, loader=default_loader):

        # load image paths
        if dataset_indicator == 'vireo':
            img_path_file = data_path + 'TR.txt'
        else:
            img_path_file = data_path + 'train_images.txt'

        with io.open(img_path_file, encoding='utf-8') as file:
            path_to_images = file.read().split('\n')[:-1]

        self.dataset_indicator = dataset_indicator
        self.image_path = image_path
        self.path_to_images = path_to_images
        self.transform = transform
        self.loader = loader
        # import ipdb; ipdb.set_trace()

    def __getitem__(self, index):
        # get image matrix and transform to tensor
        path = self.path_to_images[index]
        if self.dataset_indicator == 'vireo':
            img = self.loader(self.image_path + path)
        else:
            img = self.loader(self.image_path + path + '.jpg')

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.path_to_images)


class dataset_stage2(torch.utils.data.Dataset):
    def __init__(self, dataset_indicator, data_path=None):
        # load image paths / label file
        if True:
            # ingredients = matio.loadmat(data_path + 'ingredient_train_feature.mat')['ingredient_train_feature']
            img_path_file = data_path + 'TR.txt'
            with io.open(img_path_file, encoding='utf-8') as file:
                path_to_images = file.read().split('\n')[:-1]
            ingredients = numpy.zeros((len(path_to_images), 353))
            for i in range(len(path_to_images)):
                ingredients[i] = ingredient_library[path_to_images[i]]
            print(ingredients.shape)
            indexVectors = matio.loadmat(data_path + 'indexVector_train.mat')['indexVector_train']
        # else:
        #     ingredients = matio.loadmat(data_path + 'ingredient_all_feature.mat')['ingredient_all_feature']
        #     indexVectors = matio.loadmat(data_path + 'indexVector.mat')['indexVector']

        ingredients = ingredients.astype(np.float32)
        indexVectors = indexVectors.astype(np.long)
        self.ingredients = ingredients
        self.indexVectors = indexVectors

    def __getitem__(self, index):
        # get ingredient vector
        ingredient = self.ingredients[index, :]

        # get index vector for gru input
        indexVector = self.indexVectors[index, :]

        return [indexVector, ingredient]

    def __len__(self):
        return len(self.indexVectors)


class dataset_stage3(torch.utils.data.Dataset):
    def __init__(self, dataset_indicator, image_path=None, data_path=None, transform=None, loader=default_loader,
                 mode=None):

        # load image paths / label file
        if mode == 'train':
            if True:
                with io.open(data_path + 'TR.txt', encoding='utf-8') as file:
                    path_to_images = file.read().split('\n')[:-1]
                labels = matio.loadmat(data_path + 'train_label.mat')['train_label'][0]

                ingredients = matio.loadmat(data_path + 'ingredient_train_feature.mat')['ingredient_train_feature']
                indexVectors = matio.loadmat(data_path + 'indexVector_train.mat')['indexVector_train']
            # else:
            #     with io.open(data_path + 'train_images.txt', encoding='utf-8') as file:
            #         path_to_images = file.read().split('\n')[:-1]
            #     with io.open(data_path + 'train_labels.txt', encoding='utf-8') as file:
            #         labels = file.read().split('\n')[:-1]
            #
            #     ingredients = matio.loadmat(data_path + 'ingredient_all_feature.mat')['ingredient_all_feature']
            #     indexVectors = matio.loadmat(data_path + 'indexVector.mat')['indexVector']

        elif mode == 'test':
            if True:
                with io.open(data_path + 'TE.txt', encoding='utf-8') as file:
                    path_to_images = file.read().split('\n')[:-1]
                labels = matio.loadmat(data_path + 'test_label.mat')['test_label'][0]

                ingredients = matio.loadmat(data_path + 'ingredient_test_feature.mat')['ingredient_test_feature']
                indexVectors = matio.loadmat(data_path + 'indexVector_test.mat')['indexVector_test']
            # else:
            #     with io.open(data_path + 'test_images.txt', encoding='utf-8') as file:
            #         path_to_images = file.read().split('\n')[:-1]
            #     with io.open(data_path + 'test_labels.txt', encoding='utf-8') as file:
            #         labels = file.read().split('\n')[:-1]
            #
            #     ingredients = matio.loadmat(data_path + 'ingredient_all_feature.mat')['ingredient_all_feature']
            #     indexVectors = matio.loadmat(data_path + 'indexVector.mat')['indexVector']

        elif mode == 'val':
            if True:
                with io.open(data_path + 'VAL.txt', encoding='utf-8') as file:
                    path_to_images = file.read().split('\n')[:-1]
                labels = matio.loadmat(data_path + 'val_label.mat')['validation_label'][0]

                ingredients = matio.loadmat(data_path + 'ingredient_val_feature.mat')['ingredient_val_feature']
                indexVectors = matio.loadmat(data_path + 'indexVector_val.mat')['indexVector_val']
            # else:
            #     with io.open(data_path + 'val_images.txt', encoding='utf-8') as file:
            #         path_to_images = file.read().split('\n')[:-1]
            #     with io.open(data_path + 'val_labels.txt', encoding='utf-8') as file:
            #         labels = file.read().split('\n')[:-1]
            #
            #     ingredients = matio.loadmat(data_path + 'ingredient_all_feature.mat')['ingredient_all_feature']
            #     indexVectors = matio.loadmat(data_path + 'indexVector.mat')['indexVector']

        else:
            assert 1 < 0, 'Please fill mode with any of train/val/test to facilitate dataset creation'

        self.dataset_indicator = dataset_indicator
        self.image_path = image_path
        self.path_to_images = path_to_images
        self.labels = np.array(labels, dtype=int)

        ingredients = ingredients.astype(np.float32)
        indexVectors = indexVectors.astype(np.long)
        self.ingredients = ingredients
        self.indexVectors = indexVectors

        self.transform = transform
        self.loader = loader
        # import ipdb; ipdb.set_trace()

    def __getitem__(self, index):
        # get image matrix and transform to tensor
        path = self.path_to_images[index]
        if self.dataset_indicator == 'vireo':
            img = self.loader(self.image_path + path)
        else:
            img = self.loader(self.image_path + path + '.jpg')

        if self.transform is not None:
            img = self.transform(img)

        # get label
        label = self.labels[index]
        if self.dataset_indicator == 'food101':
            label += 1  # make labels 1-indexed to be consistent with vireo data settings
            # get ingredient vector
            ingredients = self.ingredients[label - 1]
            # get index vector for gru input
            indexVectors = self.indexVectors[label - 1]
        else:
            # get ingredient vector
            ingredients = self.ingredients[index]
            # get index vector for gru input
            indexVectors = self.indexVectors[index]

        return [img, indexVectors, ingredients, label]

    def __len__(self):
        return len(self.path_to_images)


def build_dataset(train_stage, image_path, data_path, transform, mode, dataset_indicator):
    if train_stage == 1:  # to pretrain image channel
        dataset = dataset_stage1(dataset_indicator, image_path=image_path, data_path=data_path, transform=transform)
    elif train_stage == 2:  # to pretrain ingredient channel
        dataset = dataset_stage2(dataset_indicator, data_path=data_path)
    elif train_stage == 3:  # to train the whole network
        dataset = dataset_stage3(dataset_indicator, image_path=image_path, data_path=data_path, transform=transform,
                                 mode=mode)
    else:
        assert 1 < 0, 'Please fill the correct train stage!'

    return dataset


if __name__ == '__main__':
    image_path = './ready_chinese_food/'
    data_path = './SplitAndIngreLabel/'
    mode = 'train'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    load_ingredient(data_path)
    transform_img = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    dataset = build_dataset(2, image_path, data_path, mode=mode, dataset_indicator='vireo', transform=None)
    kwargs = {}
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, **kwargs)

    print(train_loader.__len__())
