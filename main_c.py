# -*- coding: utf-8 -*-
import keras as K
import keras.layers as L
import numpy as np
import os
import random
import time
import h5py
import argparse 

from data_F import *
from models_c import *
from ops import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.layers import merge, Conv2D, MaxPool2D, Activation, Dense, concatenate, Flatten
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
import keras.backend as K1
import math
# save weights

_weights_f = "my_model_weights.h5"

_TFBooard = 'logs/events/'

parser = argparse.ArgumentParser()
parser.add_argument('--modelname', type=str,
                    default='my_model_weights.h5', help='final model save name')
parser.add_argument('--epochs',type=int,
                    default=50,help='numb er of epochs')
args = parser.parse_args()

if not os.path.exists('logs/weights/'):
    os.makedirs('logs/weights/')

if not os.path.exists(_TFBooard):
    os.mkdir(_TFBooard)





def train_merge(model):

    # # create train data
    creat_trainf(validation=False)
    creat_trainf(validation=True)






    Xh_train = np.load('./file/train_Xh.npy')
    Xh_val = np.load('./file/val_Xh.npy')
    Xl_train = np.load('./file/train_Xl.npy')
    Xl_val = np.load('./file/val_Xl.npy')



    Y_train = K.utils.np_utils.to_categorical(np.load('./file/train_Y.npy'))
    Y_val = K.utils.np_utils.to_categorical(np.load('./file/val_Y.npy'))




    print('Xl_train', Xl_train.shape)
    print('Xl_val', Xl_val.shape)
    print('Xh_train', Xh_train.shape)
    print('Xh_val', Xh_val.shape)
    print('Y_val', Y_val.shape)
    print('Y_train', Y_train.shape)



    model_ckt = ModelCheckpoint(filepath=_weights_f, monitor = 'val_loss',verbose=1, save_best_only=True)
    #

    model.fit([Xh_train, Xh_train[:, r, r, :, np.newaxis],Xl_train], Y_train, batch_size=BATCH_SIZE, epochs=args.epochs,
                     callbacks=[model_ckt], validation_data=([Xh_val, Xh_val[:, r, r, :, np.newaxis],Xl_val], Y_val))

    print(args.modelname)
    print(_weights_f)


def test(network,mode=None):
    if network == 'merge':
        model = merge_branch()
        model.load_weights(_weights_f)
        [Xl, Xh] = make_cTestf()
        pred = model.predict([Xh, Xh[:, r, r, :, np.newaxis], Xl])
        acc, kappa = cvt_map(pred, show=False)
        print('acc: {:.2f}%  Kappa: {:.4f}'.format(acc, kappa))



def main():



        model = merge_branch()
        model.summary()
        start = time.time()
        train_merge(model)
        test('merge')
        print('elapsed time:{:.2f}s'.format(time.time() - start))




if __name__ == '__main__':
    main()

