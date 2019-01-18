import numpy as np

import h5py
import cv2
import scipy.io as sio
import os
from skimage import io
from skimage.transform import resize

image_filename = "test.txt"
TARGET_PATH = os.path.join("../alldata/Labels")
IMAGE_PATH = os.path.join("../alldata/JPEGimages","")


with open(os.path.join(TARGET_PATH, image_filename), "r") as f:
    images = f.readlines()
    images_name = [name.strip("\n") for name in images]
    #print(images_name)

y = np.loadtxt(os.path.join(TARGET_PATH, "test_label"))
#print(y)

x = []

for name in images_name:
    print("reading image:" + name + ".jpg")
    img_name = os.path.join(IMAGE_PATH, name + ".jpg")

    img = cv2.imread(img_name)
    img = cv2.resize(img, (224,224))
    x.append(img)

x = np.array(x)

f = h5py.File("test_dataset.h5")
f['x'] = x
f['y'] = y
f.close()
