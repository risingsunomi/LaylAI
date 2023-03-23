from skimage.measure import block_reduce
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2

img = imread('lib/fer2013/test/angry/PrivateTest_1488292.jpg')
img = img[:,:,np.newaxis]
# img = cv2.imread('data/0/image0003155.jpg')
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = gray_img[:,:,np.newaxis]
mean_pool=block_reduce(img, block_size=(128,128,1), func=np.mean)
max_pool=block_reduce(img, block_size=(128,128,1), func=np.max)
min_pool=block_reduce(img, block_size=(128,128,1), func=np.min)
# mean_pool=block_reduce(img, block_size=(9,9,1), func=np.mean)
# max_pool=block_reduce(img, block_size=(9,9,1), func=np.max)
# min_pool=block_reduce(img, block_size=(9,9,1), func=np.min)

plt.figure(1)
plt.subplot(221)
imgplot = plt.imshow(img)
plt.title('Original Image')

plt.subplot(222)
imgplot3 = plt.imshow(mean_pool)
plt.title('Average pooling')

plt.subplot(223)
imgplot1 = plt.imshow(max_pool)
plt.title('Max pooling')

plt.subplot(224)
imgplot1 = plt.imshow(min_pool)
plt.title('Min pooling')

plt.show()