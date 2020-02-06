"""
This script is used to evaluate fine-tuned or trained models based on the appropriate validation sets.
The script needs model files (e.g. fine_tuned_model_1.h5) to be found in "tmp" folder.
The result will be created in "eval_ft_N" (or "eval_N") subfolders in "tmp" folder (N=1,2,3,4,5,6).
"""
from data import *
from model import *
import tensorflow as tf
import numpy as np
import cv2 as cv
import csv
import math


tf.compat.v1.disable_eager_execution()

for validation_set in [1, 2, 3, 4, 5, 6]:
    if validation_set < 0:
        validation_set = -validation_set
        model_path = 'trained_model_{}.h5'.format(validation_set)
        eval_folder = os.path.join(tmp_folder, 'eval_{}'.format(validation_set))
    else:
        model_path = 'fine_tuned_model_{}.h5'.format(validation_set)
        eval_folder = os.path.join(tmp_folder, 'eval_ft_{}'.format(validation_set))
    if not os.path.exists(eval_folder):
        os.mkdir(eval_folder)

    # LOADING MODEL
    model_path = os.path.join(tmp_folder, model_path)
    model = tf.keras.models.load_model(model_path, compile=False,
                                       custom_objects={'acc_fc': acc_fc,
                                                       'iou_fc': iou_fc,
                                                       'acc_iou_fc': acc_iou_fc,
                                                       'bce_dice_loss': bce_dice_loss})

    # CALCULATING PATCHES
    count = math.ceil((master_size - image_size * overlap) / (image_size * (1 - overlap)))
    step = (master_size - image_size * overlap) / count
    print('count =', count, ', step =', step)

    # CALCULATE GAUSSIAN KERNEL
    gaussian = create_gaussian()
    print("2D Gaussian kernel:")
    print(gaussian.shape)
    print(gaussian[image_size // 2, image_size // 2], gaussian[0, image_size // 2], gaussian[0, 0])

    # PROCESSING TRAINING IMAGES
    report_path = os.path.join(eval_folder, '_eval_report.csv')
    with open(report_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows([['filename', 'acc', 'iou', 'acc_iou', 'intersection', 'union', 'pixels']])
        f.flush()

        sum_intersection = 0
        sum_union = 0
        sum_pixels = 0

        for filename in src_train_images:
            name = filename[:-4]
            i = len(name) - 1
            while name[i].isdigit():
                i -= 1
            i += 1
            n = int(name[i:])

            if (n - 1) // 6 == validation_set - 1:
                img = cv.imread(os.path.join(src_train_folder, filename))
                img_gt = cv.imread(os.path.join(src_train_folder_gt, filename), 0)

                img_mask = np.zeros((master_size, master_size), np.uint8)
                output = np.zeros((master_size, master_size, 2), np.float32)

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

                img_grayscale = np.round(255 * output[..., 0] / output[..., 1]).astype(np.uint8)
                filename_grayscale = filename
                cv.imwrite(os.path.join(eval_folder, filename_grayscale), img_grayscale)

                ret, img_mask = cv.threshold(img_grayscale, round(255 * threshold), 255, cv.THRESH_BINARY)
                filename_mask = '{}_{}.png'.format(filename[:-4], threshold)
                cv.imwrite(os.path.join(eval_folder, filename_mask), img_mask)

                img_gt_mask = np.zeros((master_size, master_size, 3), np.uint8)
                img_gt_mask[..., 1] = img_gt
                img_gt_mask[..., 2] = img_mask
                img_gt_mask = np.round((0.6 * img) + (0.4 * img_gt_mask)).astype(np.uint8)
                filename_gt_mask = filename[:-4] + ".jpg"
                cv.imwrite(os.path.join(eval_folder, filename_gt_mask), img_gt_mask)

                acc = acc_img(img_gt, img_mask)
                iou = iou_img(img_gt, img_mask)
                intersection = np.sum(np.minimum(img_gt, img_mask) / 255)
                union = np.sum(np.maximum(img_gt, img_mask) / 255)
                print(filename, acc, iou, intersection, union)
                writer.writerows([[filename[:-4], acc, iou, (acc + iou) / 2, intersection, union, master_size**2]])
                f.flush()

                sum_intersection += intersection
                sum_union += union
                sum_pixels += master_size**2

        total_acc = (sum_pixels - sum_union + sum_intersection) / sum_pixels
        total_iou = sum_intersection / sum_union if sum_union > 0 else 1
        print('TOTAL', total_acc, total_iou, sum_intersection, sum_union)
        writer.writerows([['TOTAL', total_acc, total_iou, (total_acc + total_iou) / 2, sum_intersection, sum_union,
                           sum_pixels]])

    model = None
