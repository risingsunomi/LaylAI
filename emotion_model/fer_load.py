#!/usr/bin/env python3

"""
Load the FER database for tensor flow
Instead of loading from data folder

The split is done by the folders 

Right now 80% train / 20% test split
Need a way to add in changing this split in the code
Possibly by just move data while going through the folders
"""

import os
import cv2
import numpy as np
import tensorflow as tf

def load(RGB=False):
    data_path = "{}/lib/fer2013".format(os.getcwd())

    test_imgs = []
    test_categories = []

    train_imgs = []
    train_categories = []
    
    
    for folder in os.listdir(data_path):
        category_folders = os.path.join(data_path, folder)
        
        for cid, category_folder in enumerate(os.listdir(category_folders)):
            multi_class_categories = [0 for _ in os.listdir(category_folders)]
            multi_class_categories[cid] = 1

            category_folder_path = os.path.join(category_folders, category_folder)

            for img_file in os.listdir(category_folder_path):
                img_data = cv2.imread(os.path.join(category_folder_path, img_file))
                
                if not RGB:
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
                    img_data = tf.reshape(img_data, (img_data.shape[0], img_data.shape[1], 1))

                if folder == "test":
                    test_imgs.append(img_data)
                    test_categories.append(multi_class_categories)
                else:
                    train_imgs.append(img_data)
                    train_categories.append(multi_class_categories)


    print("[FER Loaded]\n\ntest_imgs: {}\ntest_categories: {}\ntrain_imgs: {}\ntrain_categories: {}\n".format(
        len(test_imgs),
        len(test_categories),
        len(train_imgs),
        len(train_categories)
    ))

    test_imgs = np.array(test_imgs)
    test_imgs = test_imgs.astype("float32") / 255

    test_categories = np.array(test_categories)

    # test_categories = tf.keras.utils.to_categorical(
    #     test_categories,
    #     num_classes=len(os.listdir(category_folders)))

    train_imgs = np.array(train_imgs)
    train_imgs = train_imgs / 255.0

    train_categories = np.array(train_categories)
    # train_categories = tf.keras.utils.to_categorical(
    #     train_categories,
    #     num_classes=len(os.listdir(category_folders)))

    tf.convert_to_tensor(test_imgs)
    tf.convert_to_tensor(test_categories)
    tf.convert_to_tensor(train_imgs)
    tf.convert_to_tensor(train_categories)

    return (test_imgs, test_categories, train_imgs, train_categories)

if __name__ == "__main__":
    load()