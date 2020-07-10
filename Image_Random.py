# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 08:47:50 2020

@author: BEL
"""

import numpy as np
from PIL import Image
import tensorflow as tf
import keras
img_data = np.random.random(size=(100, 100, 3))
img = tf.keras.preprocessing.image.array_to_img(img_data)
array = tf.keras.preprocessing.image.img_to_array(img)
