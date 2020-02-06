"""
This script is used to prepare grayscale predictions for the test images used for submission.
The script requires an ensemble of trained models that should be stored in "tmp" folder (e.g. fine_tuned_model_1.h5).
The ensemble is specified with model_name and model_numbers parameters.
The result will be created in "submission_grayscale" subfolder in "tmp" folder.
"""
from data import *
from model import *
import tensorflow as tf
import numpy as np
import cv2 as cv
import math
import time
import datetime


tf.compat.v1.disable_eager_execution()

# PARAMETERS
# model_name = 'trained_model'
model_name = 'fine_tuned_model'
model_numbers = [1, 2, 3, 4, 5, 6]


# CALCULATING PATCHES
count = math.ceil((master_size - image_size * overlap) / (image_size * (1 - overlap)))
step = (master_size - image_size * overlap) / count
print('count =', count, ', step =', step)


# CALCULATE GAUSSIAN KERNEL
gaussian = create_gaussian()
print("2D Gaussian kernel:")
print(gaussian.shape)
print(gaussian[image_size // 2, image_size // 2], gaussian[0, image_size // 2], gaussian[0, 0])


# LOADING MODELS
models = []
for model_number in model_numbers:
    print('Loading model #{}'.format(model_number))
    model_path = '{}_{}.h5'.format(model_name, model_number)
    model_path = os.path.join(tmp_folder, model_path)
    model = tf.keras.models.load_model(model_path, compile=False,
                                       custom_objects={'acc_fc': acc_fc,
                                                       'iou_fc': iou_fc,
                                                       'acc_iou_fc': acc_iou_fc,
                                                       'bce_dice_loss': bce_dice_loss})
    models.append(model)


# PROCESSING TEST IMAGES
submission_folder = os.path.join(tmp_folder, 'submission_grayscale')
if not os.path.exists(submission_folder):
    os.mkdir(submission_folder)
start_time = time.time()
image_count = len(src_test_images)
skipped = 0
for index in range(image_count):
    filename = src_test_images[index]
    if os.path.exists(os.path.join(submission_folder, filename)):
        print('Skipping image ' + filename)
        skipped += 1
    else:
        print('Processing image ' + filename)
        name = filename[:-4]
        i = len(name) - 1
        while name[i].isdigit():
            i -= 1
        i += 1
        n = int(name[i:])

        img = cv.imread(os.path.join(src_test_folder, filename))

        output = np.zeros((master_size, master_size, 2), np.float32)

        for model in models:
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

                    batch_x = np.zeros((6, image_size, image_size, 3), np.uint8)
                    batch_i = 0

                    img0 = img[y:y+image_size, x:x+image_size]
                    img1 = DataAugmentation.transform(img0.copy(), 1)
                    img2 = DataAugmentation.transform(img0.copy(), 2)
                    img3 = DataAugmentation.transform(img0.copy(), 3)
                    img4 = DataAugmentation.transform(img0.copy(), 4)
                    img5 = DataAugmentation.transform(img0.copy(), 5)

                    batch_x[batch_i] = img0
                    batch_i += 1
                    batch_x[batch_i] = img1
                    batch_i += 1
                    batch_x[batch_i] = img2
                    batch_i += 1
                    batch_x[batch_i] = img3
                    batch_i += 1
                    batch_x[batch_i] = img4
                    batch_i += 1
                    batch_x[batch_i] = img5
                    batch_i += 1

                    pred = model.predict(preprocessing(batch_x), batch_size=batch_x.shape[0])
                    pred = pred[..., 0]

                    batch_i = 0
                    mask0 = pred[batch_i]
                    batch_i += 1
                    mask1 = DataAugmentation.inverse_transform(pred[batch_i].copy(), 1)
                    batch_i += 1
                    mask2 = DataAugmentation.inverse_transform(pred[batch_i].copy(), 2)
                    batch_i += 1
                    mask3 = DataAugmentation.inverse_transform(pred[batch_i].copy(), 3)
                    batch_i += 1
                    mask4 = DataAugmentation.inverse_transform(pred[batch_i].copy(), 4)
                    batch_i += 1
                    mask5 = DataAugmentation.inverse_transform(pred[batch_i].copy(), 5)
                    batch_i += 1

                    output[y:y+image_size, x:x+image_size, 0] += mask0 * gaussian
                    output[y:y+image_size, x:x+image_size, 0] += mask1 * gaussian
                    output[y:y+image_size, x:x+image_size, 0] += mask2 * gaussian
                    output[y:y+image_size, x:x+image_size, 0] += mask3 * gaussian
                    output[y:y+image_size, x:x+image_size, 0] += mask4 * gaussian
                    output[y:y+image_size, x:x+image_size, 0] += mask5 * gaussian
                    output[y:y+image_size, x:x+image_size, 1] += 6 * gaussian

                print('.', end='')
            print()

        img_mask = np.round(255 * output[..., 0] / output[..., 1]).astype(np.uint8)
        cv.imwrite(os.path.join(submission_folder, filename), img_mask)

        elapsed_time = time.time() - start_time
        remaining_time = round(elapsed_time / (index + 1 - skipped) * (image_count - (index + 1)))
        print(str(index + 1) + '/' + str(image_count) + ' - ' + str(datetime.timedelta(seconds=remaining_time)))
