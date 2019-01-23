import os

import numpy as np
from PIL import Image
import cv2

def list_image_files(dir):
    files = sorted(os.listdir(dir))
    return [os.path.join(dir, f) for f in files]


def load_image(path):
    img = Image.open(path)
    return img


def saveImgs(A_paths, B_paths, save_path):
    all_A_paths, all_B_paths = list_image_files(A_paths), list_image_files(B_paths)
    images_A, images_B = [], []
    images_A_paths, images_B_paths = [], []
    for path_A, path_B in zip(all_A_paths, all_B_paths):
        img_A = load_image(path_A)
        img_B = cutImg(path_B)
        # images_A.append(img_A)
        # images_B.append(img_B)
        print('img:\t', path_A.split('/')[-1], ' + ', path_B.split('/')[-1] )
        # print(img_A.size)
        # print(img_B.size)
        # print('B:\t', path_B)
        img = np.concatenate((img_A, img_B), axis=0)
        im = Image.fromarray(img)
        im.save(os.path.join(save_path, path_A.split('/')[-1]))


def cutImg(img_path):
    img = cv2.imread(img_path)
    # print(img.shape)
    i = (1600-768)/2
    j = (1600-256)/2
    i = int(i*0.5)
    j = int(j*0.9)
    # print(i, '\t', j)
    # print(j, '\t', 1600-j, '\t', 1600-2*j)
    # print(i, '\t', 1600-i, '\t', 1600-2*i)
    img2 = img[j:1600 - j + 20, i:1600 - i + 40, :]
    RESHAPE = (768, 256)
    return Image.fromarray(cv2.resize(img2, RESHAPE))



if __name__=='__main__':
    saveImgs('../res/img/', '../res/img_xyy_result/', '../res/img_combine/')
    # showImg('../res/img_xyy_result/res_00_0.png', '../res/res_00_0_1.png')