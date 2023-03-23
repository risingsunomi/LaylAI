#!/usr/bin/env python3

"""
Training as multiclass with the FER dataset
"""

import tensorflow as tf
import datetime
import math
import fer_load


def lr_schedule(epoch):
    lr = 0.1
    if epoch > 10:
        lr = lr / (epoch/30)
    return lr


def step_decay(epoch):
    initial_lr = 0.001
    drop = 0.5
    epochs_drop = 30.0
    lr = initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lr


def train(RGB=False):
    # load fer data
    categories = 7
    w_img = 48
    h_img = 48
    batch_size = 512
    num_epochs = 100

    learning_rate = 0.00001
    bw_input_shape = (w_img, h_img, 1)
    rgb_input_shape = (w_img, h_img, 3)

    if not RGB:
        (test_imgs, test_categories, train_imgs, train_categories) = fer_load.load()

        # # Define the model
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Input(shape=bw_input_shape, name='input'))
        model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(
            pool_size=(5, 5), strides=(2, 2)))

        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        # model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        # model.add(tf.keras.layers.BatchNormalization())
        # model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
        # model.add(tf.keras.layers.BatchNormalization())
        # model.add(tf.keras.layers.MaxPooling2D(pool_size=(5,5), strides=(2, 2)))
        model.add(tf.keras.layers.AveragePooling2D(
            pool_size=(3, 3), strides=(2, 2)))

        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
        # model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
        # model.add(tf.keras.layers.BatchNormalization())
        # model.add(tf.keras.layers.Conv2D(128, (1,1), activation='relu'))
        # model.add(tf.keras.layers.BatchNormalization())
        # model.add(tf.keras.layers.Conv2D(128, (1,1), activation='relu'))
        # model.add(tf.keras.layers.BatchNormalization())
        # model.add(tf.keras.layers.Conv2D(128, (1,1), activation='relu'))
        # model.add(tf.keras.layers.BatchNormalization())
        # model.add(tf.keras.layers.Conv2D(128, (1,1), activation='relu'))
        # model.add(tf.keras.layers.BatchNormalization())
        # model.add(tf.keras.layers.AveragePooling2D((1,1)))
        # model.add(tf.keras.layers.MaxPooling2D((1,1)))
        model.add(tf.keras.layers.AveragePooling2D(
            pool_size=(3, 3), strides=(2, 2)))
        # model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2, 2)))
        # model.add(tf.keras.layers.Dropout(0.5))

        # model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu'))
        # model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu'))
        # model.add(tf.keras.layers.AveragePooling2D(pool_size=(3,3), strides=(1, 1)))
        # model.add(tf.keras.layers.AveragePooling2D((1,1)))
        # model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu'))
        # model.add(tf.keras.layers.MaxPooling2D((1,1)))
        # model.add(tf.keras.layers.Dropout(0.2))

        # model.add(tf.keras.layers.Conv2D(512, (3,3), activation='relu'))
        # model.add(tf.keras.layers.Conv2D(512, (3,3), activation='relu'))
        # model.add(tf.keras.layers.Conv2D(512, (3,3), activation='relu'))
        # model.add(tf.keras.layers.MaxPooling2D((1,1)))
        # model.add(tf.keras.layers.Dropout(0.2))

        # model.add(tf.keras.layers.Conv2D(512, (3,3), activation='relu'))
        # model.add(tf.keras.layers.Conv2D(512, (3,3), activation='relu'))
        # model.add(tf.keras.layers.Conv2D(512, (3,3), activation='relu'))
        # model.add(tf.keras.layers.MaxPooling2D((1,1)))
        # model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(1024, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(1024, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.3))
        # model.add(tf.keras.layers.Dense(
        #    128,
        #    activation='relu',
        #    kernel_regularizer=tf.keras.regularizers.l2(0.05)))
        # model.add(tf.keras.layers.Dense(64, activation='relu'))
        # model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(categories, activation='softmax'))

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )

        model.summary()

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)

        # learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(step_decay)

        # Train the model
        model.fit(
            train_imgs,
            train_categories,
            batch_size=batch_size,
            epochs=num_epochs,
            verbose=1,
            use_multiprocessing=True,
            validation_data=(
                test_imgs,
                test_categories),
            callbacks=[
                # learning_rate_callback,
                tensorboard_callback
            ])

        # Save the model
        model.save(
            "model/[FER]face_recognition_model__{}_{}.h5".format(
                datetime.datetime.now().strftime("%Y%m%d"),
                datetime.datetime.now().strftime("%H%M%S")
            ))

    else:
        (test_imgs, test_categories, train_imgs,
         train_categories) = fer_load.load(RGB=True)

        # -------------
        # using vgg16
        # -------------

        vgg_input = tf.keras.applications.vgg16.VGG16(
            weights="imagenet",
            include_top=False,
            input_tensor=tf.keras.layers.Input(rgb_input_shape)
        )

        vgg_input.training = False

        flatten = vgg_input.output
        flatten = tf.keras.layers.Flatten()(flatten)

        emotion_layer_1 = tf.keras.layers.Dense(
            1024, activation='relu')(flatten)
        emotion_layer_1_2 = tf.keras.layers.Dropout(0.5)(emotion_layer_1)
        emotion_layer_2 = tf.keras.layers.Dense(
            1024, activation='relu')(emotion_layer_1_2)
        emotion_layer_2_1 = tf.keras.layers.Dropout(0.5)(emotion_layer_2)
        emotion_layer_3 = tf.keras.layers.Dense(512, activation='relu')(emotion_layer_2_1)
        emotion_layer_3_1 = tf.keras.layers.Dropout(0.5)(emotion_layer_3)
        emotion_layer_4 = tf.keras.layers.Dense(256, activation='relu')(emotion_layer_3_1)
        emotion_layer_4_1 = tf.keras.layers.Dropout(0.5)(emotion_layer_4)
        emotion_layer_5 = tf.keras.layers.Dense(128, activation='relu')(emotion_layer_4_1)
        emotion_layer_5_1 = tf.keras.layers.Dropout(0.5)(emotion_layer_5)
        emotion_layer_6 = tf.keras.layers.Dense(64, activation='relu')(emotion_layer_5_1)
        emotion_layer_6_1 = tf.keras.layers.Dropout(0.2)(emotion_layer_6)
        emotion_layer_7 = tf.keras.layers.Dense(32, activation='relu')(emotion_layer_6_1)
        emotion_layer_7_1 = tf.keras.layers.Dropout(0.2)(emotion_layer_7)
        emotion_layer_8 = tf.keras.layers.Dense(16, activation='relu')(emotion_layer_7_1)
        emotion_layer_8_1 = tf.keras.layers.Dropout(0.3)(emotion_layer_8)
        emotion_layer_9 = tf.keras.layers.Dense(8, activation='relu')(emotion_layer_8_1)

        # emotion_layer_2_1 = tf.keras.layers.GlobalAveragePooling2D()(flatten)
        emotion_layer_final = tf.keras.layers.Dense(
            categories, activation='softmax')(emotion_layer_9)
        # emotion_layer_final = tf.keras.layers.Dense(
        #     categories, activation='softmax')(flatten)

        vgg_model = tf.keras.Model(
            vgg_input.input,
            outputs=emotion_layer_final)

        # Compile the model
        vgg_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        vgg_model.summary()

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)
        
        # learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(step_decay)

        # Train the model
        vgg_model.fit(
            train_imgs,
            train_categories,
            # steps_per_epoch=len(x_train) / batch_size,
            # validation_steps=len(y_train) / batch_size,
            # steps_per_epoch=2,
            # validation_steps=2,
            batch_size=batch_size,
            epochs=num_epochs,
            verbose=1,
            use_multiprocessing=True,
            validation_data=(
                test_imgs,
                test_categories),
            callbacks=[
                # learning_rate_callback,
                tensorboard_callback])

        # Save the model
        vgg_model.save(
            "model/face_recognition_model__{}_{}.h5".format(
                datetime.datetime.now().strftime("%Y%m%d"),
                datetime.datetime.now().strftime("%H%M%S")
            ))


if __name__ == "__main__":
    print("\n---- starting training for emotion model ---\n")
    train(True)
