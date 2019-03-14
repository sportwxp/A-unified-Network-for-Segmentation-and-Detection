import os
import xml.etree.ElementTree as ET

import numpy as np
from utils.config import opt
import glob
from PIL import Image

from data.util import read_image,read_mask

# IMAGE_PATH = '/home/xpwang/Documents/Data/jpg_anotation'
IMAGE_PATH = '/home/xpwang/Documents/Data/data/all_images/images'
# IMAGE_PATH = '/home/xpwang/Documents/Data/add_mask'

MASK_PATH = '/home/xpwang/Documents/Data/resoult'
class VOCBboxDataset:
    """Bounding box dataset for PASCAL `VOC`_.

    .. _`VOC`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    The index corresponds to each image.

    When queried by an index, if :obj:`return_difficult == False`,
    this dataset returns a corresponding
    :obj:`img, bbox, label`, a tuple of an image, bounding boxes and labels.
    This is the default behaviour.
    If :obj:`return_difficult == True`, this dataset returns corresponding
    :obj:`img, bbox, label, difficult`. :obj:`difficult` is a boolean array
    that indicates whether bounding boxes are labeled as difficult or not.

    The bounding boxes are packed into a two dimensional tensor of shape
    :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the
    four attributes are coordinates of the top left and the bottom right
    vertices.

    The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
    :math:`R` is the number of bounding boxes in the image.
    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`VOC_BBOX_LABEL_NAMES`.

    The array :obj:`difficult` is a one dimensional boolean array of shape
    :math:`(R,)`. :math:`R` is the number of bounding boxes in the image.
    If :obj:`use_difficult` is :obj:`False`, this array is
    a boolean array with all :obj:`False`.

    The type of the image, the bounding boxes and the labels are as follows.

    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`
    * :obj:`difficult.dtype == numpy.bool`

    Args:
        data_dir (string): Path to the root of the training data. 
            i.e. "/data/image/voc/VOCdevkit/VOC2007/"
        split ({'train', 'val', 'trainval', 'test'}): Select a split of the
            dataset. :obj:`test` split is only available for
            2007 dataset.
        year ({'2007', '2012'}): Use a dataset prepared for a challenge
            held in :obj:`year`.
        use_difficult (bool): If :obj:`True`, use images that are labeled as
            difficult in the original annotation.
        return_difficult (bool): If :obj:`True`, this dataset returns
            a boolean array
            that indicates whether bounding boxes are labeled as difficult
            or not. The default value is :obj:`False`.

    """

    def __init__(self, data_dir,
                 use_difficult=False, return_difficult=False,
                 ):

        # if split not in ['train', 'trainval', 'val']:
        #     if not (split == 'test' and year == '2007'):
        #         warnings.warn(
        #             'please pick split from \'train\', \'trainval\', \'val\''
        #             'for 2012 dataset. For 2007 dataset, you can pick \'test\''
        #             ' in addition to the above mentioned splits.'
        #         )
        #id_list_file = os.path.join(
        #   data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        self.ids = [record for record in open(data_dir,'r')]
        self.ids = [x for x in self.ids if int(x.split('|')[1].split(',')[0])!= 0 ]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = VOC_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        id_ = self.ids[i]
        filename,os_location = id_.split('|')
        xmin,xmax,ymin,ymax = [int(x) for x in os_location.split(',')]
        bbox = [[ymin,xmin,ymax,xmax]]
        label = [0]
        difficult = False

        bbox = np.array(bbox).astype(np.float32)
        label = np.array(label).astype(np.int32)
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool

        # Load a image
        img_file = os.path.join(IMAGE_PATH,filename)
        img = read_image(img_file, color=True)# (C,H,W) 255

        mask_file = os.path.join(MASK_PATH,filename)
        mask = read_mask(mask_file) # (1,H,W) 0-1.0
        # print(mask.shape)

        assert img.shape[1] == mask.shape[1]
        assert img.shape[2] == mask.shape[2]


        return img,bbox,label,difficult,mask

    __getitem__ = get_example


VOC_BBOX_LABEL_NAMES = (
    'os')
if __name__ == '__main__' :
    data = VOCBboxDataset(opt.train_data_dir)