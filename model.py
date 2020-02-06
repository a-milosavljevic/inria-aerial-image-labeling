"""
INCLUDE ONLY, DO NOT EXECUTE
"""
from settings import *
import numpy as np
import tensorflow as tf
import segmentation_models as sm


def create_model(border=False, trainable_encoder=False):
    if model_type == 'unet':
        model = sm.Unet(backbone_name=backbone,
                        input_shape=(image_size, image_size, 3),
                        classes=2 if border else 1,
                        activation='sigmoid',
                        encoder_weights='imagenet',
                        encoder_freeze=not trainable_encoder,
                        encoder_features='default',
                        decoder_block_type='upsampling',
                        decoder_filters=(256, 128, 64, 32, 16),
                        decoder_use_batchnorm=True)
    elif model_type == 'fpn':
        model = sm.FPN(backbone_name=backbone,
                       input_shape=(image_size, image_size, 3),
                       classes=2 if border else 1,
                       activation='sigmoid',
                       encoder_weights='imagenet',
                       encoder_freeze=not trainable_encoder,
                       encoder_features='default',
                       pyramid_block_filters=256,
                       pyramid_use_batchnorm=True,
                       pyramid_aggregation='concat',
                       pyramid_dropout=None)
    elif model_type == 'linknet':
        model = sm.Linknet(backbone_name=backbone,
                           input_shape=(image_size, image_size, 3),
                           classes=2 if border else 1,
                           activation='sigmoid',
                           encoder_weights='imagenet',
                           encoder_freeze=not trainable_encoder,
                           encoder_features='default',
                           decoder_block_type='upsampling',
                           decoder_filters=(None, None, None, None, 16),
                           decoder_use_batchnorm=True)
    elif model_type == 'pspnet':
        model = sm.PSPNet(backbone_name=backbone,
                          input_shape=(image_size, image_size, 3),
                          classes=2 if border else 1,
                          activation='sigmoid',
                          encoder_weights='imagenet',
                          encoder_freeze=not trainable_encoder,
                          downsample_factor=8,
                          psp_conv_filters=512,
                          psp_pooling_type='avg',
                          psp_use_batchnorm=True,
                          psp_dropout=None)
    else:
        print('Invalid segmentation model type')
        exit(0)
    return model


preprocessing = sm.get_preprocessing(backbone)
iou = sm.metrics.IOUScore(per_image=False)


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    y_true_ = y_true[..., 0]  # !!! only first channel
    y_pred_ = y_pred[..., 0]  # !!! only first channel
    dice = sm.losses.dice_loss(y_true_, y_pred_)
    return bce + dice


def iou_fc(y_true, y_pred):
    y_true_ = y_true[..., 0]  # !!! only first channel
    y_pred_ = y_pred[..., 0]  # !!! only first channel
    return iou(y_true_, y_pred_)


def acc_fc(y_true, y_pred):
    y_true_ = y_true[..., 0]  # !!! only first channel
    y_pred_ = y_pred[..., 0]  # !!! only first channel
    return tf.keras.metrics.binary_accuracy(y_true_, y_pred_)


def acc_iou_fc(y_true, y_pred):
    return (acc_fc(y_true, y_pred) + iou_fc(y_true, y_pred)) / 2


def acc_img(gt, pred):
    gt_ = np.clip(gt, 0, 1)
    pred_ = np.clip(pred, 0, 1)
    return np.mean(np.equal(gt_, pred_))


def iou_img(gt, pred):
    gt_ = np.clip(gt, 0, 1)
    pred_ = np.clip(pred, 0, 1)
    intersection = np.sum(np.minimum(gt_, pred_))
    union = np.sum(np.maximum(gt_, pred_))
    if union > 0:
        return intersection / union
    return 1
