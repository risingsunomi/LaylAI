#!/usr/bin/env python3

# training script for creating a emotion from faces model
import tensorflow as tf
import numpy as np
import os
import cv2
import datetime
import math

def lr_schedule(epoch):
  lr = 0.1
  if epoch > 10:
    lr = lr / (epoch/30)
  return lr


def step_decay(epoch):
    initial_lr = 0.00001
    drop = 0.001
    epochs_drop = 30.0
    lr = initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lr

def train():
    # Load the face dataset
    data_path = "{}/data/".format(os.getcwd())
    x_data = []
    y_data = []

    w_img = 48
    h_img = 48
    batch_size = 86
    num_epochs = 1000

    learning_rate = 0.00001
    input_shape = (w_img, h_img, 1)

    for folder in os.listdir(data_path):
        for file in os.listdir(os.path.join(data_path, folder)):
            img = cv2.imread(os.path.join(data_path, folder, file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = tf.reshape(img, (img.shape[0], img.shape[1], 1))
            x_data.append(img)
            # x_data.append(img)
            y_data.append(int(folder))

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Preprocess the data
    x_data = x_data / 255.0

    # One-hot encode the labels
    y_data = tf.keras.utils.to_categorical(y_data)

    # Split the data into training and validation sets
    x_train = x_data[:int(0.85 * len(x_data))]
    y_train = y_data[:int(0.85 * len(y_data))]
    x_val = x_data[int(0.85 * len(x_data)):]
    y_val = y_data[int(0.85* len(y_data)):]

    # Define the model
    model = tf.keras.Sequential()


    # model.add(tf.keras.layers.Conv2D(48, (3,3), activation='relu', input_shape=(w_img, h_img, 1)))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Conv2D(48, (3,3), activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.AveragePooling2D())
    # model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', input_shape=(w_img, h_img, 1)))
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(5,5), strides=(2, 2)))
    # model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    # model.add(tf.keras.layers.Dropout(0.3))

    
    # model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    # model.add(tf.keras.layers.Dropout(0.3))
    
    
    # model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    # model.add(tf.keras.layers.Dropout(0.3))
    # model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    # model.add(tf.keras.layers.AveragePooling2D())


    # model.add(tf.keras.layers.Flatten())

    # , kernel_regularizer=tf.keras.regularizers.l2(0.05)
    # model.add(tf.keras.layers.Dense(1024, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.3))
    # model.add(tf.keras.layers.Dense(1024, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.3))
    
    # model.add(tf.keras.layers.Dense(1024, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(512, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(64, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(32, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(16, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))

    # model.add(tf.keras.layers.Dense(
    #     len(os.listdir(data_path)),
    #     activation='softmax'))


    
    # model.add(tf.keras.layers.Conv2D(16, (3,3), 1, activation='relu', input_shape=(w_img, h_img, 3)))
    # model.add(tf.keras.layers.MaxPooling2D())
    # model.add(tf.keras.layers.Conv2D(32, (3,3), 1, activation='relu'))
    # model.add(tf.keras.layers.MaxPooling2D())
    # model.add(tf.keras.layers.Conv2D(16, (3,3), 1, activation='relu'))
    # model.add(tf.keras.layers.MaxPooling2D())

    # model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(w_img, h_img, 3)))
    # model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    # model.add(tf.keras.layers.Dropout(0.3))
    # model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    # model.add(tf.keras.layers.Dropout(0.3))
    # model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    # model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    # model.add(tf.keras.layers.Dropout(0.3))
    # model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
    # model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    # model.add(tf.keras.layers.Dropout(0.3))
    # model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
    # model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    # model.add(tf.keras.layers.Dropout(0.3))

    # model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(w_img, h_img, 1)))
    # model.add(tf.keras.layers.AveragePooling2D(pool_size=(3,3), strides=(2, 2)))
    # model.add(tf.keras.layers.Dropout(0.5))

    # model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu',  padding='same', input_shape=(w_img, h_img, 1)))
    # model.add(tf.keras.layers.AveragePooling2D(pool_size=(5,5), strides=(2, 2)))

    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Conv2D(64, (3, 3),  padding='same', activation='relu'))
    # model.add(tf.keras.layers.Conv2D(64, (3, 3),  padding='same', activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    # model.add(tf.keras.layers.AveragePooling2D(pool_size=(3,3), strides=(2, 2)))
    # model.add(tf.keras.layers.Dropout(0.5))
    

    # model.add(tf.keras.layers.Conv2D(128, (3, 3),  padding='same', activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Conv2D(128, (3, 3),  padding='same', activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    # model.add(tf.keras.layers.AveragePooling2D(pool_size=(3,3), strides=(2, 2)))
    # model.add(tf.keras.layers.Dropout(0.5))

    # model.add(tf.keras.layers.Conv2D(256, (3, 3),  padding='same', activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Conv2D(256, (3, 3),  padding='same', activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    # model.add(tf.keras.layers.Dropout(0.3))
    # model.add(tf.keras.layers.AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

    # model.add(tf.keras.layers.Conv2D(256, (3, 3),  padding='same', activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Conv2D(256, (3, 3),  padding='same', activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    # model.add(tf.keras.layers.Dropout(0.3))
    # model.add(tf.keras.layers.AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

    # model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    # model.add(tf.keras.layers.Dropout(0.3))

    

    # model.add(tf.keras.layers.Dense(256, activation='relu'))
    # model.add(tf.keras.layers.Dense(1024, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.2))
    # model.add(tf.keras.layers.Dense(1024, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.2))

    # model.add(tf.keras.layers.Dense(128, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(64, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(32, activation='relu'))

    # model.add(tf.keras.layers.Dense(128, activation='relu',
                                # kernel_regularizer=tf.keras.regularizers.l2(0.05)))
    # model.add(tf.keras.layers.Dense(64, activation='relu'))
    # model.add(tf.keras.layers.Dense(32, activation='relu'))


    model.add(tf.keras.layers.Input(shape=input_shape, name='input'))
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((1,1)))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu'))
    model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((1,1)))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu'))
    model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu'))
    model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((1,1)))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(512, (3,3), activation='relu'))
    model.add(tf.keras.layers.Conv2D(512, (3,3), activation='relu'))
    model.add(tf.keras.layers.Conv2D(512, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((1,1)))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(512, (3,3), activation='relu'))
    model.add(tf.keras.layers.Conv2D(512, (3,3), activation='relu'))
    model.add(tf.keras.layers.Conv2D(512, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((1,1)))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(len(os.listdir(data_path)), activation='softmax'))

    # Compile the model

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )

    model.summary()

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(step_decay)

    # Train the model
    model.fit(
        tf.convert_to_tensor(x_train),
        tf.convert_to_tensor(y_train),
        batch_size=batch_size,
        epochs=num_epochs,
        verbose=1,
        use_multiprocessing=True,
        validation_data=(
            tf.convert_to_tensor(x_val), 
            tf.convert_to_tensor(y_val)),
        callbacks=[
            # learning_rate_callback, 
            tensorboard_callback
        ])

    # Save the model
    model.save(
        "model/face_recognition_model__{}_{}.h5".format(
        datetime.datetime.now().strftime("%Y%m%d"),
        datetime.datetime.now().strftime("%H%M%S")
    ))

    # -------------
    # using vgg16
    # -------------
    # does not change much just slower
    # vgg_input = tf.keras.applications.vgg16.VGG16(
    #     weights="imagenet",
    #     include_top=False,
    #     input_tensor= tf.keras.layers.Input((
    #         w_img,
    #         h_img,
    #         3
    #     )) 
    # )

    # vgg_input.training = False

    # flatten = vgg_input.output
    # flatten = tf.keras.layers.Flatten()(flatten)

    # emotion_layer_1 = tf.keras.layers.Dense(1024, activation='relu')(flatten)
    # emotion_layer_1_2 = tf.keras.layers.Dropout(0.2)(emotion_layer_1)
    # emotion_layer_2 = tf.keras.layers.Dense(1024, activation='relu')(emotion_layer_1_2)
    # emotion_layer_2_1 = tf.keras.layers.Dropout(0.2)(emotion_layer_2) 

    # emotion_layer_2_1 = tf.keras.layers.GlobalAveragePooling2D()(flatten)  
    # emotion_layer_final = tf.keras.layers.Dense(len(os.listdir(data_path)), activation='softmax')(emotion_layer_2_1)

    # vgg_model = tf.keras.Model(
    #     vgg_input.input,
    #     outputs=emotion_layer_final)

    # # Compile the model
    # vgg_model.compile(
    #     optimizer=tf.keras.optimizers.Adam(),#learning_rate=learning_rate), 
    #     loss='categorical_crossentropy', 
    #     metrics=['accuracy']
    # )

    # vgg_model.summary()

    # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Train the model
    # vgg_model.fit(
    #     tf.convert_to_tensor(x_train),
    #     tf.convert_to_tensor(y_train),
    #     # steps_per_epoch=len(x_train) / batch_size,
    #     # validation_steps=len(y_train) / batch_size,
    #     # steps_per_epoch=2,
    #     # validation_steps=2,
    #     epochs=num_epochs,
    #     verbose=1,
    #     use_multiprocessing=True,
    #     validation_data=(
    #         tf.convert_to_tensor(x_val), 
    #         tf.convert_to_tensor(y_val)),
    #     callbacks=[tensorboard_callback])

    # # Save the model
    # vgg_model.save(
    #     "model/face_recognition_model__{}_{}.h5".format(
    #     datetime.datetime.now().strftime("%Y%m%d"),
    #     datetime.datetime.now().strftime("%H%M%S")
    # ))

    # # model from
    # input_shape = (w_img, h_img, 1)
    # visible = tf.keras.layers.Input(shape=input_shape, name='input')
    
    # #the 1-st block
    # conv1_1 = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same', name = 'conv1_1')(visible)
    # conv1_1 = tf.keras.layers.BatchNormalization()(conv1_1)
    # conv1_2 = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same', name = 'conv1_2')(conv1_1)
    # conv1_2 = tf.keras.layers.BatchNormalization()(conv1_2)
    # pool1_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), name = 'pool1_1')(conv1_2)

    # drop1_1 = tf.keras.layers.Dropout(0.5, name = 'drop1_1')(pool1_1)#the 2-nd block
    # conv2_1 = tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_1')(drop1_1)
    # conv2_1 = tf.keras.layers.BatchNormalization()(conv2_1)
    # conv2_2 = tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_2')(conv2_1)
    # conv2_2 = tf.keras.layers.BatchNormalization()(conv2_2)
    # conv2_3 = tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_3')(conv2_2)
    # conv2_2 = tf.keras.layers.BatchNormalization()(conv2_3)
    # pool2_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), name = 'pool2_1')(conv2_3)
    # drop2_1 = tf.keras.layers.Dropout(0.5, name = 'drop2_1')(pool2_1)#the 3-rd block

    # conv3_1 = tf.keras.layers.Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_1')(drop2_1)
    # conv3_1 = tf.keras.layers.BatchNormalization()(conv3_1)
    # conv3_2 = tf.keras.layers.Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_2')(conv3_1)
    # conv3_2 = tf.keras.layers.BatchNormalization()(conv3_2)
    # conv3_3 = tf.keras.layers.Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_3')(conv3_2)
    # conv3_3 = tf.keras.layers.BatchNormalization()(conv3_3)
    # conv3_4 = tf.keras.layers.Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_4')(conv3_3)
    # conv3_4 = tf.keras.layers.BatchNormalization()(conv3_4)
    # pool3_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), name = 'pool3_1')(conv3_4)
    # drop3_1 = tf.keras.layers.Dropout(0.5, name = 'drop3_1')(pool3_1)#the 4-th block

    # conv4_1 = tf.keras.layers.Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_1')(drop3_1)
    # conv4_1 = tf.keras.layers.BatchNormalization()(conv4_1)
    # conv4_2 = tf.keras.layers.Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_2')(conv4_1)
    # conv4_2 = tf.keras.layers.BatchNormalization()(conv4_2)
    # conv4_3 = tf.keras.layers.Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_3')(conv4_2)
    # conv4_3 = tf.keras.layers.BatchNormalization()(conv4_3)
    # conv4_4 = tf.keras.layers.Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_4')(conv4_3)
    # conv4_4 = tf.keras.layers.BatchNormalization()(conv4_4)
    # pool4_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), name = 'pool4_1')(conv4_4)
    # drop4_1 = tf.keras.layers.Dropout(0.5, name = 'drop4_1')(pool4_1)
    
    # #the 5-th block
    # conv5_1 = tf.keras.layers.Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_1')(drop4_1)
    # conv5_1 = tf.keras.layers.BatchNormalization()(conv5_1)
    # conv5_2 = tf.keras.layers.Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_2')(conv5_1)
    # conv5_2 = tf.keras.layers.BatchNormalization()(conv5_2)
    # conv5_3 = tf.keras.layers.Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_3')(conv5_2)
    # conv5_3 = tf.keras.layers.BatchNormalization()(conv5_3)
    # conv5_4 = tf.keras.layers.Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_4')(conv5_3)
    # conv5_3 = tf.keras.layers.BatchNormalization()(conv5_3)
    # pool5_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), name = 'pool5_1')(conv5_4)
    # drop5_1 = tf.keras.layers.Dropout(0.5, name = 'drop5_1')(pool5_1)#Flatten and output
    # flatten = tf.keras.layers.Flatten(name = 'flatten')(drop5_1)
    # output = tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.08))(flatten)
    # output_2 = tf.keras.layers.Dense(len(os.listdir(data_path)), activation='softmax', name = 'output')(output)# create model 
    # model = tf.keras.Model(visible, outputs=output_2)

    # # Compile the model
    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
    #     loss='categorical_crossentropy', 
    #     metrics=['accuracy']
    # )

    # model.summary()

    # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # # Train the model
    # model.fit(
    #     tf.convert_to_tensor(x_train),
    #     tf.convert_to_tensor(y_train),
    #     # steps_per_epoch=len(x_train) / batch_size,
    #     batch_size=batch_size,
    #     epochs=num_epochs,
    #     verbose=1,
    #     use_multiprocessing=True,
    #     validation_data=(
    #         tf.convert_to_tensor(x_val), 
    #         tf.convert_to_tensor(y_val)),
    #     callbacks=[tensorboard_callback])

    # # Save the model
    # model.save(
    #     "model/face_recognition_model__{}_{}.h5".format(
    #     datetime.datetime.now().strftime("%Y%m%d"),
    #     datetime.datetime.now().strftime("%H%M%S")
    # ))

if __name__ == "__main__":
    print("\n---- starting training for emotion model ---\n")
    train()