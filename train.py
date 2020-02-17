"""
Execute this script to run the initial training phase using binary_crossentropy loss.
In case of an out of memory problem adjust batch_size in settings.py.
Be sure to run prepare_data.py first.
"""
from data import *
from model import *
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np


tf.compat.v1.disable_eager_execution()


for validation_set in [1, 2, 3, 4, 5, 6]:
    ####################################################################################################################
    # MODEL
    ####################################################################################################################

    model_path = 'trained_model_{}.h5'.format(validation_set)
    model_path = os.path.join(tmp_folder, model_path)

    training_log_path = 'training_log_{}.csv'.format(validation_set)
    training_log_path = os.path.join(tmp_folder, training_log_path)

    model = create_model(border=border, trainable_encoder=True)

    optimizer = tf.keras.optimizers.RMSprop(lr=init_lr)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=[acc_fc, iou_fc, acc_iou_fc])

    model.summary()

    ####################################################################################################################
    # CALLBACKS
    ####################################################################################################################

    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True,
                                                      monitor=cb_monitor[0], mode=cb_monitor[1])
    csv_logger = tf.keras.callbacks.CSVLogger(training_log_path, separator=',', append=False)

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=early_stopping_patience, restore_best_weights=True,
                                                      verbose=1, monitor=cb_monitor[0], mode=cb_monitor[1])

    def cosine_annealing_schedule(t, lr, period=lr_period, scale=lr_scale, decay=lr_decay):
        step = t // period
        lr = init_lr * (decay**step)
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

    model.fit(data_gen_train,
              epochs=epochs,
              validation_data=data_gen_valid,
              shuffle=True,
              callbacks=[checkpointer, csv_logger, early_stopping, schedule_lr],
              verbose=1)

    model.save(model_path, include_optimizer=False)
    model = None
    K.clear_session()
