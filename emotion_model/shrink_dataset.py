#!/usr/bin/env python3

# shrinks down dataset samples due to ram limits
# setup naming convention of folders to be fed into tensorflow

import os
import shutil
import cv2
import numpy as np


def shrink():
    data_path = "{}/lib/affectnet_hq".format(os.getcwd())
    dist_path = "{}/data".format(os.getcwd())
    folder_num = 0

    for folder in os.listdir(data_path):
        # Path to the large dataset
        dataset_path = os.path.join(data_path, folder)

        # Number of samples per folder
        samples_per_folder = 4000

        # Get a list of all the files in the dataset
        files = os.listdir(dataset_path)

        # Split the files into smaller chunks of size 'samples_per_folder'
        # split_files = [files[i:i + samples_per_folder] for i in range(0, len(files), samples_per_folder)]

        # Loop through each chunk and create a folder for it
        for file in files:
            if samples_per_folder > 0:
                file_path = os.path.join(dataset_path, file)

                # Load the cascade classifier
                # face_cascade = cv2.CascadeClassifier('{}/lib/haarcascade_frontalface_default.xml'.format(os.getcwd()))

                # Load the input image
                img = cv2.imread(file_path)
                small_img = cv2.resize(img, (48, 48))

                # # Detect faces in the grayscale image
                # faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

                # # Crop and save the faces
                # for (x, y, w, h) in faces:
                #     face_only = img[y:y + h, x:x + w]
                    
                folder_path = os.path.join(dist_path, str(folder_num))
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)

                # face_img = os.path.join(folder_path, "face_{}".format(file))

                # face_only_gray = cv2.cvtColor(face_only, cv2.COLOR_BGR2GRAY)
                # face_only_gray = cv2.resize(face_only_gray, (64, 64))

                cv2.imwrite(os.path.join(folder_path, file), small_img)

                # Copy the files in this chunk to the newly created folder
                # shutil.copy(file_path, folder_path)
            
            samples_per_folder -= 1
        
        folder_num += 1


if __name__ == "__main__":
    shrink()