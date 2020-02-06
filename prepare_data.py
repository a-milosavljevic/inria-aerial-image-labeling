"""
This script is used to create image and ground truth patches from original data files.
Before executing this script be sure to:
1) Fill in the form and download data from https://project.inria.fr/aerialimagelabeling/download/
2) Extract downloaded files and place them into "data" folder using the following folder structure:
   data/test/images/*.tif
   data/train/images/*.tif
   data/train/gt/*.tif
"""
from data import *
import numpy as np
import cv2 as cv
import shutil
import math


test = False

count = math.ceil((master_size - image_size * overlap) / (image_size * (1 - overlap)))
step = (master_size - image_size * overlap) / count
print('count =', count, ', step =', step)

if test:
    img = np.zeros((master_size, master_size), np.uint8)

    for i in range(count):
        if i < count - 1:
            y = round(i * step)
        else:
            y = master_size - image_size

        for j in range(count):
            if j < count - 1:
                x = round(j * step)
            else:
                x = master_size - image_size
            cv.rectangle(img, pt1=(x, y), pt2=(x + image_size - 1, y + image_size - 1), color=255)

    cv.imwrite(os.path.join(tmp_folder, "test.png"), img)
    exit(0)

if not os.path.exists(train_folder_root):
    os.makedirs(train_folder_root)

if not os.path.exists(train_folder):
    os.makedirs(train_folder)
else:
    shutil.rmtree(train_folder)

if not os.path.exists(train_folder_gt):
    os.makedirs(train_folder_gt)
else:
    shutil.rmtree(train_folder_gt)

for filename in src_train_images:
    print(filename)
    master_img = cv.imread(os.path.join(src_train_folder, filename))
    master_img_gt = cv.imread(os.path.join(src_train_folder_gt, filename))

    for i in range(count):
        if i < count - 1:
            y = round(i * step)
        else:
            y = master_size - image_size

        for j in range(count):
            if j < count - 1:
                x = round(j * step)
            else:
                x = master_size - image_size

            img = master_img[y:y+image_size, x:x+image_size]
            img_gt = master_img_gt[y:y+image_size, x:x+image_size]

            img_fname = '{}_{}_{}.{}'.format(filename[:-4], i, j, 'jpg')
            img_gt_fname = '{}_{}_{}.{}'.format(filename[:-4], i, j, 'png')
            cv.imwrite(os.path.join(train_folder, img_fname), img)
            cv.imwrite(os.path.join(train_folder_gt, img_gt_fname), img_gt)
