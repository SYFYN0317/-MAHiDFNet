# -*- coding: utf-8 -*-
from __future__ import print_function, division
import tensorflow as tf
import keras as K
import keras.layers as L
import numpy as np
import os
import time
import h5py
import argparse
import matplotlib.pyplot as plt
from attention import *
from data_F import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import Input, Activation, Conv2D, Dropout
from keras.layers import MaxPooling2D, BatchNormalization
from keras.layers import UpSampling2D
from keras.layers import concatenate
from keras.layers import add
from attention import PAM, DPAM
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Activation,Layer
import keras.backend as K1
from cross import *
from keras.optimizers import RMSprop
# ===================cascade net=============

def attention_block_3(inputs,feature_cnt,dim):
    a = Flatten()(inputs)
    a = Dense(feature_cnt*dim,activation='softmax')(a)
    a = L.Reshape((feature_cnt,dim,))(a)
    a = L.Lambda(lambda x: K1.sum(x, axis=2), name='attention')(a)
    a = L.RepeatVector(dim)(a)
    a_probs = L.Permute((2, 1), name='attention_vec')(a)
    attention_out = L.Multiply()([inputs, a_probs])
    return attention_out

def small_cnn_branch(input_tensor, small_mode=True):
    filters=[32,64,100,200,256]
    conv0_spat=L.Conv2D(filters[2],(3,3),padding='same')(input_tensor)
    conv0_spat=L.BatchNormalization(axis=-1)(conv0_spat)
    conv0_spat=L.advanced_activations.LeakyReLU(alpha=0.2)(conv0_spat)
    conv1_spat=L.Conv2D(filters[2],(3,3),padding='same')(conv0_spat)
    conv1_spat=L.BatchNormalization(axis=-1)(conv1_spat)
    conv1_spat=L.advanced_activations.LeakyReLU(alpha=0.2)(conv1_spat)
    conv2_spat=L.Conv2D(filters[3],(1,1),padding='same')(conv1_spat)
    conv2_spat=L.BatchNormalization(axis=-1)(conv2_spat)
    conv2_spat=L.advanced_activations.LeakyReLU(alpha=0.2)(conv2_spat)
    conv3_spat=L.Conv2D(filters[3],(1,1),padding='same')(conv2_spat)
    conv3_spat=L.BatchNormalization(axis=-1)(conv3_spat)
    conv3_spat=L.advanced_activations.LeakyReLU(alpha=0.2)(conv3_spat)
    conv3_spat = PAM()(conv3_spat)
    pool3=L.MaxPool2D(pool_size=(2,2),padding='same')(conv3_spat)
    Dense1=L.Dense(1024)(pool3)
    Dense1=L.Activation('relu')(Dense1)
    Dense1=L.Dropout(0.5)(Dense1)
    Dense2=L.Dense(512)(Dense1)
    Dense2=L.Activation('relu')(Dense2)
    Dense2=L.Dropout(0.5)(Dense2)
    conv7_spat=L.Flatten()(Dense2)
    return conv7_spat




def small_cnn_branch_front(input_tensor):
    filters=[32,64,100,200,256]
    conv0_spat=L.Conv2D(filters[2],(3,3),padding='same')(input_tensor)
    conv0_spat=L.BatchNormalization(axis=-1)(conv0_spat)
    conv0_spat=L.advanced_activations.LeakyReLU(alpha=0.2)(conv0_spat)
    conv1_spat=L.Conv2D(filters[2],(3,3),padding='same')(conv0_spat)
    conv1_spat=L.BatchNormalization(axis=-1)(conv1_spat)
    conv1_spat=L.advanced_activations.LeakyReLU(alpha=0.2)(conv1_spat)
    conv2_spat=L.Conv2D(filters[3],(1,1),padding='same')(conv1_spat)
    conv2_spat=L.BatchNormalization(axis=-1)(conv2_spat)
    conv2_spat=L.advanced_activations.LeakyReLU(alpha=0.2)(conv2_spat)
    conv3_spat=L.Conv2D(filters[3],(1,1),padding='same')(conv2_spat)
    conv3_spat=L.BatchNormalization(axis=-1)(conv3_spat)
    conv3_spat=L.advanced_activations.LeakyReLU(alpha=0.2)(conv3_spat)
    conv3_spat=L.Conv2D(filters[3],(1,1),padding='same')(conv3_spat)
    conv3_spat=L.BatchNormalization(axis=-1)(conv3_spat)
    conv3_spat=L.advanced_activations.LeakyReLU(alpha=0.2)(conv3_spat)

    return conv3_spat


def small_cnn_branch_latter(input_tensor):
    pool1=L.MaxPool2D(pool_size=(2,2),padding='same')(input_tensor)
    Dense1=L.Dense(1024)(pool1)
    Dense1=L.Activation('relu')(Dense1)
    Dense1=L.Dropout(0.4)(Dense1)
    Dense2=L.Dense(512)(Dense1)
    Dense2=L.Activation('relu')(Dense2)
    Dense2=L.Dropout(0.4)(Dense2)
    conv7_spat=L.Flatten()(Dense2)
    return conv7_spat





def pixel_branch(input_tensor):
    filters = [8, 16, 32, 64, 96, 128]
    conv0 = L.Conv1D(filters[3], 11, padding='valid')(input_tensor)
    conv0_a = attention_block_3(conv0,170,64)
    conv0 = L.concatenate([conv0,conv0_a])
    conv0 = L.BatchNormalization(axis=-1)(conv0)
    conv0 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv0)
    conv3 = L.Conv1D(filters[5], 3, padding='valid')(conv0)
    conv3 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv3)
    conv3 = L.MaxPool1D(pool_size=2, padding='valid')(conv3)
    conv3 = L.Conv1D(filters[5], 3, padding='valid')(conv3)
    conv3 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv3)
    conv3 = L.MaxPool1D(pool_size=2, padding='valid')(conv3)
    conv3 = L.Conv1D(filters[5], 3, padding='valid')(conv3)
    conv3 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv3)
    conv3 = L.MaxPool1D(pool_size=2, padding='valid')(conv3)
    conv3 = L.Flatten()(conv3)
    return conv3



def merge_branch():
    ksize = 2 * r + 1
    filters = [64, 128, 256, 512]
    hsi_in = L.Input((ksize, ksize, hchn))
    hsi_pxin = L.Input((hchn, 1))
    lidar_in = L.Input((ksize, ksize,lchn))

    h_simple = small_cnn_branch(hsi_in, small_mode=False)
    px_out = pixel_branch(hsi_pxin)


    ha_simple = small_cnn_branch_front(hsi_in)
    la_simple = small_cnn_branch_front(lidar_in)

    ha_simple_c = L.Lambda(cross)([la_simple,ha_simple])
    la_simple_c = L.Lambda(cross)([ha_simple,la_simple])

    ha_simple = DPAM()(ha_simple_c)
    la_simple = DPAM()(la_simple_c)

    ha_simple = small_cnn_branch_latter(ha_simple)
    la_simple = small_cnn_branch_latter(la_simple)

    merge1=L.concatenate([h_simple,px_out], axis=-1)
    merge1=L.Dropout(0.5)(merge1)
    merge2=L.concatenate([ha_simple,la_simple], axis=-1)
    merge2=L.Dropout(0.5)(merge2)
    merge=L.concatenate([merge1,merge2], axis=-1)
    merge=L.Dropout(0.5)(merge)

    logits = L.Dense(NUM_CLASS, activation='softmax',name='logits_out')(merge)


    model = K.models.Model([hsi_in,hsi_pxin,lidar_in], logits)
    adam = K.optimizers.Adam(lr=0.0001,beta_1=0.9,beta_2=0.999,epsilon=1e-8)
    optm = K.optimizers.SGD(lr=0.00005,momentum=1e-6,nesterov=True)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy', metrics=['acc'])


    return model

