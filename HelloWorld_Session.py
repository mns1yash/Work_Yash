# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:28:05 2020

@author: BEL
"""


import tensorflow as tf
tf.compat.v1.disable_eager_execution()
hello = tf.constant('Hello  Tensorworld !')
sess = tf.compat.v1.Session()
print(sess.run(hello))
sess.close()



# tf.config.experimental.list_physical_devices()
# tf.test.is_gpu_available()
