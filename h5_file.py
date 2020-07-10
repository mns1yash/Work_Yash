# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 12:17:42 2020

@author: BEL
"""
import tensorflow as tf
import keras
import h5py
f = h5py.File('individual_model.h5','r')
list(f.keys())
strategy = tf.distribute.MirroredStrategy()
strategy
