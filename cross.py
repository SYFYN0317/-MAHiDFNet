import numpy as np
import tensorflow as tf
import keras.layers as L


def cross(args):
    a = args[0]
    b = args[1]
    a_new = tf.expand_dims(a,-1)
    b_new = tf.expand_dims(b,-1)
    output = L.concatenate([a_new,b_new], axis=-1)
    print(output.shape)

    return output