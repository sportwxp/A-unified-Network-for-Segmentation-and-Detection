from __future__ import  absolute_import
from __future__ import  division
import torch as t
from data.voc_dataset import VOCBboxDataset
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from data import util
import numpy as np
from utils.config import opt


def inverse_normalize(img):
    # if opt.caffe_pretrain:
    #     img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
    #     return img[::-1, :, :]
    # approximate un-normalize for visualize
    mean = np.array([0.66620953, 0.48074919, 0.5001002 ]).reshape(3, 1, 1)
    std = np.array([0.18702547, 0.17880133, 0.18710214]).reshape(3, 1, 1)

    return (img * std + mean).clip(min=0, max=1) * 255


def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=[0.66620953, 0.48074919, 0.5001002 ],
                                std=[0.18702547, 0.17880133, 0.18710214])
    img = normalize(t.from_numpy(img))
    return img.numpy()


def preprocess(img,mask):
    '''
    :param img: (3,676,780),255
    :param mask: (676,780)
    :return: img: (3,512,512),-1~1
    '''
    C, H, W = img.shape
    img = img / 255.
    img = sktsf.resize(img, (C,512,512), mode='reflect',anti_aliasing=False)
    img = pytorch_normalze(img)


    mask = sktsf.resize(mask, (1,512,512), mode='reflect',anti_aliasing=False)
    return img ,mask

def preprocess_image(img):
    '''
    :param img: (3,676,780),255
    :param mask: (676,780)
    :return: img: (3,512,512),-1~1
    '''
    C, H, W = img.shape
    img = img / 255.
    img = sktsf.resize(img, (C,512,512), mode='reflect',anti_aliasing=False)
    img = pytorch_normalze(img)



    return img




class Transform(object):

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label ,mask= in_data
        _, H, W = img.shape

        img,mask = preprocess(img,mask)

        _, o_H, o_W = img.shape
        scale = o_H / H
        # print(bbox)
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        # img, params = util.random_flip(
        #     img, x_random=True, return_param=True)
        # bbox = util.flip_bbox(
        #     bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale, mask


class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = VOCBboxDataset(opt.train_data_dir)
        self.tsf = Transform()

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult,mask = self.db.get_example(idx)

        img, bbox, label, scale,mask = self.tsf((ori_img, bbox, label,mask))
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        return img.copy(), bbox.copy(), label.copy(), scale, mask.copy()

    def __len__(self):
        return len(self.db)


class TestDataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = VOCBboxDataset(self.opt.test_data_dir)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult,mask = self.db.get_example(idx)
        img,mask = preprocess(ori_img,mask)
        return img, ori_img.shape[1:], bbox, label, difficult,mask

    def __len__(self):
        return len(self.db)


if __name__ == '__main__' :
    train_data = TestDataset(opt)
    print(train_data[9])
