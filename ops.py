import numpy as np
import os
import time
import h5py
import keras as K
import keras.layers as L
from data_F import *
from models_c import *
import matplotlib.pyplot as plt

def read_image(filename):
    img = tiff.imread(filename)
    img = np.asarray(img, dtype=np.float32)
    return img

def countX(lst,x):
    count = 0
    for ele in lst:
        if(ele == x):
            count = count+1
    return count

def cvt_map(pred, show=False):
    """
    convert prediction percent to map
    """
    # gth = tiff.imread(os.path.join(PATH, gth_test))
    gth = read_mat(PATH, gth_test, TestImage)
    # gth=read_mat(PATH,gth_test,'mask_test')
    pred = np.argmax(pred, axis=1)
    pred = np.asarray(pred, dtype=np.int8) + 1
    print(pred)
    np.save(os.path.join(SAVA_PATH, 'predops.npy'), pred)
    index = np.load(os.path.join(SAVA_PATH, 'index.npy'))
    pred_map = np.zeros_like(gth)
    cls = []
    for i in range(index.shape[1]):
        pred_map[index[0, i], index[1, i]] = pred[i]
        cls.append(gth[index[0, i], index[1, i]])
    cls = np.asarray(cls, dtype=np.int8)
    if show:
        plt.imshow(pred_map)
        plt.figure()
        plt.imshow(gth)
        plt.show()
    tiff.imsave('results.tif',pred_map)
    count = np.sum(pred == cls)
    mx = confusion(pred - 1, cls - 1)
    print(mx)
    acc = 100.0 * count / np.sum(gth != 0)
    kappa = compute_Kappa(mx)
    return acc, kappa


def confusion(pred, labels):
    """
    make confusion matrix 
    """
    mx = np.zeros((NUM_CLASS, NUM_CLASS))
    if len(pred.shape) == 2:
        pred = np.asarray(np.argmax(pred, axis=1))

    for i in range(labels.shape[0]):
        mx[pred[i], labels[i]] += 1
    mx = np.asarray(mx, dtype=np.int16)
    np.savetxt('confusion.txt', mx, delimiter=" ", fmt="%s")
    return mx

def compute_Kappa(confusion_matrix):
    """
    TODO =_= 
    """
    N = np.sum(confusion_matrix)
    N_observed = np.trace(confusion_matrix)
    Po = 1.0 * N_observed / N
    h_sum = np.sum(confusion_matrix, axis=0)
    v_sum = np.sum(confusion_matrix, axis=1)
    Pe = np.sum(np.multiply(1.0 * h_sum / N, 1.0 * v_sum / N))
    kappa = (Po - Pe) / (1.0 - Pe)
    return kappa

def eval(pred, gth, show=False):
    """
    evaluate between prediction and ground truth 
    return the over accuracy
    """
    pred = np.argmax(pred, 4)
    h, w = gth.shape
    if not h % ksize == 0:
        hm = ((h // ksize) + 1) * ksize
    if not w % ksize == 0:
        wm = ((w // ksize) + 1) * ksize
    new_map = np.zeros(shape=(hm, wm))
    for i in range(pred.shape[1]):
        for j in range(pred.shape[0]):
            new_map[i * ksize:(i + 1) * ksize, j *
                    ksize:(j + 1) * ksize] = pred[j, i, :, :]
    new_map = np.asarray(new_map, dtype=np.int8)
    new_map = new_map[0:h, 0:w]
    cls_gth = np.zeros_like(gth)
    cls_map = np.zeros_like(new_map)
    cls_map[new_map != 0] = new_map[new_map != 0]
    cls_gth[gth != 0] = gth[gth != 0]
    count = np.sum(cls_gth == cls_map)
    acc = 1.0 * count / np.sum(gth != 0)
    if show:
        plt.imshow(new_map)
        plt.figure()
        plt.imshow(gth)
        plt.show()
    return acc

def visual_model(model,imgname):
    from keras.utils import plot_model
    # plot_model(model, to_file=imgname, show_shapes=True)


    
