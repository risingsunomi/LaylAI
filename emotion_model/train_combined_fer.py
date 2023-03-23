#!/usr/bin/env python3

"""
Training as multiclass with the FER dataset combing custom with VGG16
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


def train():
    # load fer data
    categories = 7
    w_img = 48
    h_img = 48
    batch_size = 256
    num_epochs = 5000

    learning_rate = 1e-9
    rgb_input_shape = (w_img, h_img, 3)

    (test_imgs, test_categories, train_imgs, train_categories) = fer_load.load(RGB=True)

    # -------------
    # vgg16 model
    # -------------
    vgg_model = tf.keras.applications.vgg16.VGG16(
            weights="imagenet",
            include_top=False,
            input_tensor=tf.keras.layers.Input(rgb_input_shape)
        )

    vgg_model.training = False

    

    # -----------------
    # 2nd vgg16 model
    # -----------------
    vgg2_model = tf.keras.applications.vgg16.VGG16(
            weights="imagenet",
            include_top=False,
            input_tensor=tf.keras.layers.Input(rgb_input_shape)
        )

    vgg2_model.training = False

    
    # -------------
    # custom model
    # -------------

    # custom_model = tf.keras.Sequential()
    # custom_model.add(tf.keras.layers.Input(
    #     shape=rgb_input_shape, name='custom_model_input'))

    # custom_model.add(
    #     tf.keras.layers.Conv2D(
    #         64, 
    #         (5, 5), 
    #         activation='relu',  
    #         padding='same', 
    #         input_shape=(w_img, h_img, 1),
    #         name='custom_cnn_1'
    # ))
    # custom_model.add(tf.keras.layers.MaxPooling2D(
    #     pool_size=(5,5),
    #     strides=(2, 2),
    #     name='custom_maxpool_1'))
    # custom_model.add(tf.keras.layers.Conv2D(
    #     64,
    #     (3, 3), 
    #     padding='same',
    #     activation='relu',
    #     name='custom_cnn_2'))
    # custom_model.add(tf.keras.layers.Conv2D(
    #     64,
    #     (3, 3), 
    #     padding='same',
    #     activation='relu',
    #     name='custom_cnn_3'))
    # custom_model.add(tf.keras.layers.AveragePooling2D(
    #     pool_size=(3,3),
    #     strides=(2, 2),
    #     name='custom_avgpool_1'))
    # custom_model.add(tf.keras.layers.Conv2D(
    #     128,
    #     (3, 3), 
    #     padding='same',
    #     activation='relu',
    #     name='custom_cnn_4'))
    # custom_model.add(tf.keras.layers.Conv2D(
    #     128, 
    #     (3, 3), 
    #     padding='same', 
    #     activation='relu',
    #     name='custom_cnn_5'))
    # custom_model.add(tf.keras.layers.AveragePooling2D(
    #     pool_size=(3,3), 
    #     strides=(2, 2),
    #     name='custom_avgpool_2'))
    

    #-----------------
    # ensemble models
    #-----------------

    ensemble_input = tf.keras.layers.Input(shape=rgb_input_shape, name='ensemble_input')
    
    # output dense layers

    vgg_output = vgg_model(ensemble_input)
    vgg_output = tf.keras.layers.Flatten(name='ensemble_flatten_vgg')(vgg_output)

    vgg_output = tf.keras.layers.Dense(
        2048, 
        activation='relu',
        name='vgg_output_dense_1',
        kernel_regularizer=tf.keras.regularizers.l2(0.1))(vgg_output)
    vgg_output = tf.keras.layers.Dense(
        1024, 
        activation='relu',
        name='vgg_output_dense_2',
        kernel_regularizer=tf.keras.regularizers.l2(0.1))(vgg_output)

    vgg2_output = vgg_model(ensemble_input)
    vgg2_output = tf.keras.layers.Flatten(name='ensemble_flatten_vgg2')(vgg2_output)

    vgg2_output = tf.keras.layers.Dense(
        2048, 
        activation='relu',
        name='vgg2_output_dense_1',
        kernel_regularizer=tf.keras.regularizers.l2(0.1))(vgg2_output)
    vgg2_output = tf.keras.layers.Dense(
        1024, 
        activation='relu',
        name='vgg2_output_dense_2',
        kernel_regularizer=tf.keras.regularizers.l2(0.1))(vgg2_output)


    
    
    # custom_output = custom_model(ensemble_input)
    # custom_output = tf.keras.layers.Flatten(name='custom_output_flatten')(custom_output)
    # custom_output = tf.keras.layers.Dense(
    #     1024, 
    #     activation='relu',
    #     name='custom_output_dense_1',
    #     kernel_regularizer=tf.keras.regularizers.l2(0.08))(custom_output)
    # custom_output = tf.keras.layers.Dropout(
    #     0.8,
    #     name='custom_output_dropout_1')(custom_output)
    # custom_output = tf.keras.layers.Dense(
    #     1024, 
    #     activation='relu',
    #     name='custom_output_dense_2',
    #     kernel_regularizer=tf.keras.regularizers.l2(0.08))(custom_output)
    # custom_output = tf.keras.layers.Dropout(
    #     0.8,
    #     name='custom_output_dropout_2')(custom_output)

    


    merged_output = tf.keras.layers.Concatenate(
        name='merged_output')([vgg_output, vgg2_output]) #custom_output])
    
    ensemble_output = tf.keras.layers.Dense(
        categories, 
        activation='softmax', 
        name='ensemble_output')(merged_output)
    
    ensemble_model = tf.keras.Model(inputs=ensemble_input, outputs=ensemble_output)
    
    ensemble_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )

    ensemble_model.summary()

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)

    # learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(step_decay)

    # Train the model
    ensemble_model.fit(
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
    ensemble_model.save(
        "model/[FERENS]face_recognition_model__{}_{}.h5".format(
            datetime.datetime.now().strftime("%Y%m%d"),
            datetime.datetime.now().strftime("%H%M%S")
        ))

if __name__ == "__main__":
    print("\n---- starting training for emotion model ---\n")
    train()