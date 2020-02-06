"""
INCLUDE ONLY, DO NOT EXECUTE
"""
import os


########################################################################################################################
# IMAGE PREPARATION PARAMETERS
########################################################################################################################

master_size = 5000
image_size = 384
overlap = 0.3
threshold = 0.45


########################################################################################################################
# TRAINING PARAMETERS
########################################################################################################################

batch_size = 9
init_lr = 1e-3
init_lr_ft = 1e-5
lr_scale = 0.01
lr_period = 5
lr_decay = 0.7
epochs = 100 * lr_period
cb_monitor = ['val_acc_iou_fc', 'max']
border = True


########################################################################################################################
# SETUP SEGMENTATION MODELS
########################################################################################################################

# model_type = 'unet'
model_type = 'fpn'
# model_type = 'linknet'
# model_type = 'pspnet'

# backbone = 'vgg16'
# backbone = 'vgg19'
# backbone = 'resnet18'
# backbone = 'resnet34'
# backbone = 'resnet50'
# backbone = 'resnet101'
# backbone = 'resnet152'
backbone = 'resnext50'
# backbone = 'resnext101'
# backbone = 'inceptionv3'
# backbone = 'inceptionresnetv2'
# backbone = 'densenet121'
# backbone = 'densenet169'
# backbone = 'densenet201'
# backbone = 'seresnet18'
# backbone = 'seresnet34'
# backbone = 'seresnet50'
# backbone = 'seresnet101'
# backbone = 'seresnet152'
# backbone = 'seresnext50'
# backbone = 'seresnext101'
# backbone = 'senet154'
# backbone = 'mobilenet'
# backbone = 'mobilenetv2'


########################################################################################################################
# SETUP FOLDERS
########################################################################################################################

root_folder = os.getcwd()

data_folder = os.path.join(root_folder, 'data')

tmp_folder = os.path.join(root_folder, 'tmp')
if not os.path.exists(tmp_folder):
    os.mkdir(tmp_folder)
