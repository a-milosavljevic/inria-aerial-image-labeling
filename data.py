"""
INCLUDE ONLY, DO NOT EXECUTE
"""
from settings import *
import numpy as np
from tensorflow.keras.utils import Sequence
import cv2 as cv


src_train_folder = os.path.join(data_folder, 'train', 'images')
src_train_folder_gt = os.path.join(data_folder, 'train', 'gt')
src_test_folder = os.path.join(data_folder, 'test', 'images')

src_train_images = os.listdir(src_train_folder)
src_test_images = os.listdir(src_test_folder)

train_folder_root = os.path.join(data_folder, 'train_{}x{}'.format(image_size, image_size))
train_folder = os.path.join(train_folder_root, 'images')
train_folder_gt = os.path.join(train_folder_root, 'gt')


def create_gaussian(size=image_size, sigma=0.55):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    d = np.sqrt(x * x + y * y)
    gaussian = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
    return gaussian


debug_folder = os.path.join(tmp_folder, 'debug')


class DataAugmentation(Sequence):

    def __init__(self, batch_size, validation, validation_set, process_input, border, debug=False):
        assert(0 <= validation_set <= 6)
        self.batch_size = batch_size
        self.validation = validation
        self.validation_set = validation_set
        self.process_input = process_input
        self.border = border
        self.debug = debug

        if self.debug:
            if not os.path.exists(debug_folder):
                os.makedirs(debug_folder)

        # Build image list
        self.images = []
        for fname in os.listdir(train_folder):
            name = fname.split('_')[0]
            i = len(name) - 1
            while name[i].isdigit():
                i -= 1
            i += 1
            n = int(name[i:])
            if validation_set > 0:
                if self.validation:
                    if (n - 1) // 6 == self.validation_set - 1:
                        self.images.append(fname)
                else:
                    if (n - 1) // 6 != self.validation_set - 1:
                        self.images.append(fname)
            elif not self.validation:
                self.images.append(fname)

        # Shuffle data
        if self.validation:
            self.images = np.random.RandomState(0).permutation(self.images)
            print("validation_elements = " + str(len(self.images)))
        else:
            self.images = np.random.RandomState(0).permutation(self.images)
            print("training_elements = " + str(len(self.images)))

        # Create border structuring element
        if self.border:
            self.structuring_element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = min(len(self.images), (idx + 1) * self.batch_size)
        batch_images = self.images[batch_start:batch_end]

        batch_x = np.zeros((len(batch_images), image_size, image_size, 3), dtype=np.float32)
        if self.border:
            batch_y = np.zeros((len(batch_images), image_size, image_size, 2), dtype=np.float32)
        else:
            batch_y = np.zeros((len(batch_images), image_size, image_size, 1), dtype=np.float32)

        for i in range(len(batch_images)):
            fname = batch_images[i]
            fpath = os.path.join(train_folder, fname)
            fpath_gt = os.path.join(train_folder_gt, fname[:-4] + '.png')
            image = cv.imread(fpath)
            image_gt = cv.imread(fpath_gt, 0)
            image_gt = np.expand_dims(image_gt, -1)

            if not self.validation:
                t = self.get_random_transform()
                image = self.transform(image, t)
                image_gt = self.transform(image_gt, t)

            batch_x[i] = self.process_input(image)

            if self.border:
                border = cv.dilate(image_gt, self.structuring_element) - cv.erode(image_gt, self.structuring_element)
                border = np.reshape(border, (image_size, image_size, 1))
                batch_y[i] = np.concatenate((image_gt, border), axis=-1) / 255
            else:
                batch_y[i] = image_gt / 255

            if self.debug:
                cv.imwrite(os.path.join(debug_folder, fname), image)
                cv.imwrite(os.path.join(debug_folder, fname[:-4] + '.png'), image_gt)
                if self.border:
                    cv.imwrite(os.path.join(debug_folder, fname[:-4] + '_b.png'), border)

        return batch_x, batch_y

    @staticmethod
    def get_random_transform():
        tc = 6
        t = min(tc-1, int(np.floor(tc * np.random.rand())))
        return t

    @staticmethod
    def transform(img, t):
        if t == 1:
            return np.fliplr(img)
        if t == 2:
            return np.flipud(img)
        if t == 3:
            return np.rot90(img, 2)
        if t == 4:
            return np.rot90(img, -1)
        if t == 5:
            return np.rot90(img, 1)
        return img

    @staticmethod
    def inverse_transform(img, t):
        if t == 1:
            return np.fliplr(img)
        if t == 2:
            return np.flipud(img)
        if t == 3:
            return np.rot90(img, -2)
        if t == 4:
            return np.rot90(img, 1)
        if t == 5:
            return np.rot90(img, -1)
        return img


def test_data_augmentation():
    img = cv.imread(os.path.join(train_folder, 'austin1_9_0.jpg'))
    for i in range(100):
        t = DataAugmentation.get_random_transform()
        img_aug = DataAugmentation.transform(img, t)
        img_aug = np.clip(img_aug, 0, 255).astype(np.uint8)
        cv.imwrite(os.path.join(tmp_folder, str(i) + '.jpg'), img_aug)


#test_data_augmentation()
