"""
Once the first training phase is completed (train.py), use this script to fine-tune the models.
The script needs previously trained models (e.g. trained_model_1.h5) to be found in "tmp" folder.
In case of an out of memory problem, adjust batch_size in settings.py.
"""
from data import *
from model import *
import tensorflow as tf
import numpy as np


tf.compat.v1.disable_eager_execution()

for validation_set in [1, 2, 3, 4, 5, 6]:
    ####################################################################################################################
    # MODEL
    ####################################################################################################################

    model_path = 'fine_tuned_model_{}.h5'.format(validation_set)
    model_path = os.path.join(tmp_folder, model_path)

    training_log_path = 'fine_tuning_log_{}.csv'.format(validation_set)
    training_log_path = os.path.join(tmp_folder, training_log_path)

    previous_model_path = 'trained_model_{}.h5'.format(validation_set)
    previous_model_path = os.path.join(tmp_folder, previous_model_path)

    model = tf.keras.models.load_model(previous_model_path, compile=False,
                                       custom_objects={'acc_fc': acc_fc,
                                                       'iou_fc': iou_fc,
                                                       'acc_iou_fc': acc_iou_fc,
                                                       'bce_dice_loss': bce_dice_loss})

    for layer in model.layers:
        layer.trainable = True

    optimizer = tf.keras.optimizers.RMSprop(lr=init_lr_ft)

    model.compile(optimizer=optimizer,
                  loss=bce_dice_loss,
                  metrics=[acc_fc, iou_fc, acc_iou_fc])

    model.summary()

    ####################################################################################################################
    # CALLBACKS
    ####################################################################################################################

    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True,
                                                      monitor=cb_monitor[0], mode=cb_monitor[1])

    csv_logger = tf.keras.callbacks.CSVLogger(training_log_path, separator=',', append=False)

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=1*lr_period, restore_best_weights=True, verbose=1,
                                                      monitor=cb_monitor[0], mode=cb_monitor[1])

    def cosine_annealing_schedule(t, lr, period=lr_period, scale=lr_scale, decay=lr_decay):
        step = t // period
        lr = init_lr_ft * (decay**step)
        arg = np.pi * (t % period) / period
        k = (np.cos(arg) + 1) / 2
        k = k * (1 - scale) + scale
        ret = float(lr * k)
        print('cosine_annealing_schedule(t={}, lr={}, period={}, scale={}, decay={}) -> {}'
              .format(t, lr, period, scale, decay, ret))
        return ret

    schedule_lr = tf.keras.callbacks.LearningRateScheduler(cosine_annealing_schedule, verbose=0)

    ####################################################################################################################
    # TRAINING
    ####################################################################################################################

    data_gen_train = DataAugmentation(batch_size, validation=False, validation_set=validation_set,
                                      process_input=preprocessing, border=border)
    data_gen_valid = DataAugmentation(batch_size, validation=True, validation_set=validation_set,
                                      process_input=preprocessing, border=border)

    model.fit_generator(data_gen_train,
                        epochs=epochs,
                        validation_data=data_gen_valid,
                        shuffle=True,
                        callbacks=[checkpointer, csv_logger, early_stopping, schedule_lr],
                        verbose=1)

    model.save(model_path, include_optimizer=False)
    model = None
