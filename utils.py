import os

import numpy as np
import torch
from PIL import Image, ImageFile
from Dataloader import DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils import data

RESHAPE = (256, 256)

def is_an_image_file(filename):
    IMAGE_EXTENSIONS = ['.png']
    for ext in IMAGE_EXTENSIONS:
        if ext in filename:
            return True
    return False


def list_image_files(dir):
    files = sorted(os.listdir(dir))
    return [os.path.join(dir, f) for f in files if is_an_image_file(f)]


def load_image(path):
    img = Image.open(path)
    return img


def preprocess_image(cv_img):
    cv_img = cv_img.resize(RESHAPE)
    img = np.array(cv_img)
    img = (img - 127.5) / 127.5
    return img


def deprocess_image(img):
    img = img * 127.5 + 127.5
    return img.astype('uint8')


def save_image(np_arr, path):
    img = np_arr * 127.5 +127.5
    img = img.transpose(1, 2, 0).astype('uint8')
    im = Image.fromarray(img)
    im.save(path)


def load_images(path, n_images):
    if n_images < 0:
        n_images = float("inf")
    A_paths, B_paths = os.path.join(path, 'A'), os.path.join(path, 'B')
    all_A_paths, all_B_paths = list_image_files(A_paths), list_image_files(B_paths)
    images_A, images_B = [], []
    images_A_paths, images_B_paths = [], []
    for path_A, path_B in zip(all_A_paths, all_B_paths):
        img_A, img_B = load_image(path_A), load_image(path_B)
        images_A.append(preprocess_image(img_A))
        images_B.append(preprocess_image(img_B))
        images_A_paths.append(path_A)
        images_B_paths.append(path_B)
        if len(images_A) > n_images - 1: break
    #
    # return {
    #     'A': np.array(images_A),
    #     'A_paths': np.array(images_A_paths),
    #     'B': np.array(images_B),
    #     'B_paths': np.array(images_B_paths)
    # }
    return {
        # 'A': torch.from_numpy(np.array(images_A).transpose(0, 3, 1, 2)),
        'A': np.array(images_A).transpose(0, 3, 1, 2),
        'A_paths': np.array(images_A_paths),
        # 'B': torch.from_numpy(np.array(images_B).transpose(0, 3, 1, 2)),
        'B': np.array(images_B).transpose(0, 3, 1, 2),
        'B_paths': np.array(images_B_paths)
    }

class MyDataset(data.Dataset):
    def __init__(self, images, targets):
        self.images = images
        self.targets = targets

    def __getitem__(self, index):
        img, target = self.images[index], self.targets[index]
        # img2 = 'img'
        img, target = torch.from_numpy(img),  torch.from_numpy(target)
        # return img, target
        return img.cuda(), target.cuda()

    def __len__(self):
        return len(self.images)


class MyDataset3(data.Dataset):
    def __init__(self, images, targets, path):
        self.images = images
        self.targets = targets
        self.path = path

    def __getitem__(self, index):
        img, target, path = self.images[index], self.targets[index], self.path[index]
        # img2 = 'img'
        img, target = torch.from_numpy(img),  torch.from_numpy(target)
        # return img, target
        return img.cuda(), target.cuda(), path

    def __len__(self):
        return len(self.images)

def test_dataset(data):
    batchSize = 2
    y_test, x_test = data['B'], data['A']
    x_path = data['A_paths']
    # print(y_train.shape)
    dataSet = MyDataset3(x_test, y_test,x_path)
    loader = DataLoader()
    loader.initialize(dataSet, batchSize, shuffle=False)
    dataset = loader.load_data()
    return dataset


def test_img(g, index, dir, dataset):
    for step, (x, y, path) in enumerate(dataset):
        img = g(x)

        if not os.path.exists(os.path.join(dir, 'imgs')):
            os.makedirs(os.path.join(dir, 'imgs'))
        # print(img.shape)
        # print(y.shape)
        res = np.concatenate((img.data.cpu().numpy(), x.data.cpu().numpy(), y.data.cpu().numpy()), axis=3)
        for i in range(x.shape[0]):
            # print(res[i].shape)
            # print(path[i])
            img_name = str(path[i]).split('/')[-1].split('.')[0]
            save_path = os.path.join(dir, 'imgs/res{}_{}.png'.format(img_name, index))
            print(save_path, ' saved!!')
            save_image(res[i], save_path)


if __name__ =='__main__':
    testData = load_images('/devdata/xyy2/MyDeblur/DeblurGan/images/test', -1)
    testDataset = test_dataset(testData)
    # path = os.path.join('E:\Code\MyDeblur\DeblurGan','images','train')
    # A_paths = os.path.join(path, 'A')
    # ls = list_image_files(A_paths)
    # data = load_images(path, n_images=100)
    # y_train, x_train = data['B'], data['A']
    # # y_train = deprocess_image(y_train)
    # # print(y_train[0].transpose(1, 2, 0).shape)
    # # print(type(y_train[0]))
    # # im = Image.fromarray(y_train[0])
    # # im.save('res.png')
    # save_image(y_train[0], 'res0.png')
