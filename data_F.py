# -*- coding: utf-8 -*-
import numpy as np
import tifffile as tiff
import os
import cv2
import scipy.io as sio
from keras.utils.np_utils import to_categorical
from scipy.cluster.vq import whiten


NUM_CLASS = 7
PATH = './data/Aug'
SAVA_PATH = './file/'
BATCH_SIZE = 100
r = 5

HSName='data_HS_LR.mat'#height_test.tif'
data_HS_LR ='data_HS_LR'
lidarName='data_DSM.mat'#height_test.tif'
data_DSM ='data_DSM'
gth_train = 'TrainImage.mat'
TrainImage ='TrainImage'
gth_test = 'TestImage.mat'
TestImage = 'TestImage'
lchn = 1
hchn = 180

# NUM_CLASS = 15
# PATH = './data/houston'
# SAVA_PATH = './file/'
# BATCH_SIZE = 100
# r = 5
#
#
# HSName='Houston_HS.tif'#height_test.tif'
# lidarName='Houston_Lidar.tif'#height_test.tif'
# gth_train = 'Houston_train.tif'
# gth_test = 'Houston_test.tif'
# lchn = 1
# hchn = 144

# NUM_CLASS = 11
# PATH = './data/MUUFL'
# SAVA_PATH = './file/'
# BATCH_SIZE = 100
# r = 5
#
#
# HSName='MUUFL_HS.tif'#height_test.tif'
# lidarName='MUUFL_Lidar.tif'#height_test.tif'
# gth_train = 'MUUFL_train.tif'
# gth_test = 'MUUFL_test.tif'
# lchn = 2
# hchn = 64

# NUM_CLASS = 6
# PATH = './data/Trento'
# SAVA_PATH = './file/'
# BATCH_SIZE = 64
# r = 5
#
#
# HSName='Trento_HSI.tif'#height_test.tif'
# lidarName='Trento_Lidar.tif'#height_test.tif'
# gth_train = 'Trento_train.tif'
# gth_test = 'Trento_test.tif'
# lchn = 1
# hchn = 63



if not os.path.exists(SAVA_PATH):
    os.mkdir(SAVA_PATH)


def read_image(filename):
    img = tiff.imread(filename)
    img = np.asarray(img, dtype=np.float32)
    return img

def read_mat(path,file_name,data_name):
    mdata=sio.loadmat(os.path.join(path,file_name))
    mdata=np.array(mdata[data_name])
    return mdata


def samele_wise_normalization(data):
    """
    normalize each sample to 0-1
    Input:
        sample
    Output:
        Normalized sample
    """
    if np.max(data) == np.min(data):
        return np.ones_like(data, dtype=np.float32) * 1e-6
    else:
        return 1.0 * (data - np.min(data)) / (np.max(data) - np.min(data))


def sample_wise_standardization(data):
    import math
    _mean = np.mean(data)
    _std = np.std(data)
    npixel = np.size(data) * 1.0
    min_stddev = 1.0 / math.sqrt(npixel)
    return (data - _mean) / max(_std, min_stddev)


def gth2mask(gth):
    # gth[gth>7]-=1
    # gth-=1
    new_gth = np.zeros(
        shape=(gth.shape[0], gth.shape[1], NUM_CLASS), dtype=np.int8)
    for c in range(NUM_CLASS):
        new_gth[gth == c, c] = 1
    return new_gth

def down_sampling_hsi(hsi, scale=2):
    hsi = cv2.GaussianBlur(hsi, (3, 3), 0)
    hsi = cv2.resize(cv2.resize(hsi,
                                (hsi.shape[1] // scale, hsi.shape[0] // scale),
                                interpolation=cv2.INTER_CUBIC),
                     (hsi.shape[1], hsi.shape[0]),
                     interpolation=cv2.INTER_CUBIC)
    return hsi

def creat_trainf(validation=False):
    hsi = read_mat(PATH,HSName,data_HS_LR)
    lidar = read_mat(PATH,lidarName,data_DSM)
    gth = read_mat(PATH,gth_train,TrainImage)
    # hsi = read_image(os.path.join(PATH, HSName))
    # lidar = read_image(os.path.join(PATH, lidarName))
    # gth = tiff.imread(os.path.join(PATH, gth_train))
    hsi = np.pad(hsi, ((r, r), (r, r), (0, 0)), 'symmetric')
    if len(lidar.shape) == 2:
        lidar = np.pad(lidar, ((r, r), (r, r)), 'symmetric')
    if len(lidar.shape) == 3:
        lidar = np.pad(lidar, ((r, r), (r, r), (0, 0)), 'symmetric')
    gth = np.pad(gth, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    # gth = np.pad(gth, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    per = 0.89


    # lidar = samele_wise_normalization(lidar)
    lidar = sample_wise_standardization(lidar)
    # hsi = samele_wise_normalization(hsi)
    hsi = sample_wise_standardization(hsi)
    # hsi=whiten(hsi)

    Xh = []
    Xl = []
    Y = []
    for c in range(1, NUM_CLASS + 1):
        idx, idy = np.where(gth == c)
        if not validation:
            idx = idx[:int(per * len(idx))]
            idy = idy[:int(per * len(idy))]
        else:
            idx = idx[int(per * len(idx)):]
            idy = idy[int(per * len(idy)):]
        np.random.seed(820)
        ID = np.random.permutation(len(idx))
        idx = idx[ID]
        idy = idy[ID]
        for i in range(len(idx)):
            tmph = hsi[idx[i] - r:idx[i] + r + 1, idy[i] - r:idy[i] + r + 1, :]
            tmpl = lidar[idx[i] - r:idx[i] + r +
                         1, idy[i] - r:idy[i] + r + 1]
            tmpy = gth[idx[i], idy[i]] - 1
            Xh.append(tmph)
            Xh.append(np.flip(tmph, axis=0))
            noise = np.random.normal(0.0, 0.01, size=tmph.shape)
            Xh.append(np.flip(tmph + noise, axis=1))
            k = np.random.randint(4)
            Xh.append(np.rot90(tmph, k=k))


            Xl.append(tmpl)
            Xl.append(np.flip(tmpl, axis=0))
            noise = np.random.normal(0.0, 0.03, size=tmpl.shape)
            Xl.append(np.flip(tmpl + noise, axis=1))
            Xl.append(np.rot90(tmpl, k=k))




            Y.append(tmpy)
            Y.append(tmpy)
            Y.append(tmpy)
            Y.append(tmpy)


    index = np.random.permutation(len(Xh))
    Xh = np.asarray(Xh, dtype=np.float32)
    Xl = np.asarray(Xl, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.int8)
    Xh = Xh[index, ...]
    if len(Xl.shape) == 3:
        Xl = Xl[index, ..., np.newaxis]
    elif len(Xl.shape) == 4:
        Xl = Xl[index, ...]
    Y = Y[index]
    print('train hsi data shape:{},train lidar data shape:{}'.format(
        Xh.shape, Xl.shape))
    if not validation:
        np.save(os.path.join(SAVA_PATH, 'train_Xh.npy'), Xh)
        np.save(os.path.join(SAVA_PATH, 'train_Xl.npy'), Xl)
        np.save(os.path.join(SAVA_PATH, 'train_Y.npy'), Y)
    else:
        np.save(os.path.join(SAVA_PATH, 'val_Xh.npy'), Xh)
        np.save(os.path.join(SAVA_PATH, 'val_Xl.npy'), Xl)
        np.save(os.path.join(SAVA_PATH, 'val_Y.npy'), Y)

def make_AL():
    hsi = read_image(os.path.join(PATH, HSName))
    lidar = read_image(os.path.join(PATH, lidarName))
    gthTR = tiff.imread(os.path.join(PATH, gth_train))
    gthTE = tiff.imread(os.path.join(PATH, gth_test))

    gth =gthTE-gthTR
    hsi = np.pad(hsi, ((r, r), (r, r), (0, 0)), 'symmetric')
    if len(lidar.shape) == 2:
        lidar = np.pad(lidar, ((r, r), (r, r)), 'symmetric')
    if len(lidar.shape) == 3:
        lidar = np.pad(lidar, ((r, r), (r, r), (0, 0)), 'symmetric')
    gth = np.pad(gth, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    # gth=read_mat(PATH,gth_test,'mask_test')

    # lidar = samele_wise_normalization(lidar)
    lidar = sample_wise_standardization(lidar)
    # hsi = samele_wise_normalization(hsi)
    hsi = sample_wise_standardization(hsi)
    # hsi=whiten(hsi)
    idx, idy = np.where(gth != 0)
    ID = np.random.permutation(len(idx))
    Xh = []
    Xl = []
    for i in range(len(idx)):
        tmph = hsi[idx[ID[i]] - r:idx[ID[i]] + r +
                   1, idy[ID[i]] - r:idy[ID[i]] + r + 1, :]
        tmpl = lidar[idx[i] - r:idx[i] + r +
                                1, idy[i] - r:idy[i] + r + 1]
        Xh.append(tmph)
        Xl.append(tmpl)
    Xh = np.asarray(Xh, dtype=np.float32)
    Xl = np.asarray(Xl, dtype=np.float32)
    if len(Xl.shape) == 3:
        Xl = Xl[..., np.newaxis]
    # print index
    np.save(os.path.join(SAVA_PATH, 'hsiAL.npy'), Xh)
    np.save(os.path.join(SAVA_PATH, 'lidarAL.npy'), Xl)
    np.save(os.path.join(SAVA_PATH, 'indexAL.npy'), [idx[ID] - r, idy[ID] - r])
    return  Xh,Xl

def make_cTestf():
    HS = read_mat(PATH,HSName,data_HS_LR)
    lidar = read_mat(PATH,lidarName,data_DSM)
    gth = read_mat(PATH,gth_test,TestImage)
    # lidar = read_image(os.path.join(PATH, lidarName))
    # HS = read_image(os.path.join(PATH, HSName))
    # gth = tiff.imread(os.path.join(PATH, gth_test))

    HS = np.pad(HS, ((r, r), (r, r), (0, 0)), 'symmetric')
    if len(lidar.shape) == 2:
        lidar = np.pad(lidar, ((r, r), (r, r)), 'symmetric')
    if len(lidar.shape) == 3:
        lidar = np.pad(lidar, ((r, r), (r, r), (0, 0)), 'symmetric')
    gth = np.pad(gth, ((r, r), (r, r)), 'constant', constant_values=(0, 0))

    # lidar = samele_wise_normalization(lidar)
    lidar = sample_wise_standardization(lidar)
    # HS = samele_wise_normalization(HS)
    HS = sample_wise_standardization(HS)
    # hsi=whiten(hsi)
    idx, idy = np.where(gth != 0)
    np.random.seed(820)
    ID = np.random.permutation(len(idx))
    Xh = []
    Xl = []
    for i in range(len(idx)):
        tmph = HS[idx[ID[i]] - r:idx[ID[i]] + r +
                   1, idy[ID[i]] - r:idy[ID[i]] + r + 1, :]
        tmpl = lidar[idx[ID[i]] - r:idx[ID[i]] +
                     r + 1, idy[ID[i]] - r:idy[ID[i]] + r + 1]

        Xh.append(tmph)
        Xl.append(tmpl)
    Xh = np.asarray(Xh, dtype=np.float32)
    Xl = np.asarray(Xl, dtype=np.float32)
    index = np.concatenate(
        (idx[..., np.newaxis], idy[..., np.newaxis]), axis=1)
    np.save(os.path.join(SAVA_PATH, 'hsi.npy'), Xh)
    np.save(os.path.join(SAVA_PATH, 'lidar.npy'), Xl)
    np.save(os.path.join(SAVA_PATH, 'index.npy'), [idx[ID] - r, idy[ID] - r])
    if len(Xl.shape) == 3:
        Xl = Xl[..., np.newaxis]
    return Xl, Xh
