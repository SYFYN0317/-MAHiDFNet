from keras.layers import Activation, Conv2D
import keras.backend as K1
import tensorflow as tf
from keras.layers import Layer


class PAM(Layer):
    def __init__(self,
                 # gamma_initializer=tf.zeros_initializer(),
                 # gamma_regularizer=None,
                 # gamma_constraint=None,
                 **kwargs):

        super(PAM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1, ),
                                     initializer='zeros',
                                     regularizer=None,
                                     constraint=None,
                                     name='gamma',
                                     trainable=True
                                     )

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input):
        input_shape = input.get_shape().as_list()
        _, h, w, filters = input_shape

        b = Conv2D(filters // 8, 1,use_bias=False, kernel_initializer='he_normal' )(input)
        c = Conv2D(filters // 8, 1,use_bias=False, kernel_initializer='he_normal')(input)
        d = Conv2D(filters, 1,use_bias=False, kernel_initializer='he_normal')(input)

        vec_b = K1.reshape(b, (-1, h * w, filters // 8))
        vec_cT = tf.transpose(K1.reshape(c, (-1, h * w, filters // 8)), (0, 2, 1))
        bcT = K1.batch_dot(vec_b, vec_cT)
        softmax_bcT = Activation('softmax')(bcT)
        vec_d = K1.reshape(d, (-1, h * w, filters))
        bcTd = K1.batch_dot(softmax_bcT, vec_d)
        bcTd = K1.reshape(bcTd, (-1, h, w, filters))

        out = self.gamma*bcTd + input
        return out



class DPAM(Layer):
    def __init__(self,
                 # gamma_initializer=tf.zeros_initializer(),
                 # gamma_regularizer=None,
                 # gamma_constraint=None,
                 **kwargs):
        # self.gamma_initializer = gamma_initializer
        # self.gamma_regularizer = gamma_regularizer
        # self.gamma_constraint = gamma_constraint
        super(DPAM, self).__init__(**kwargs)

    def build(self, input_shape):

        self.gamma = self.add_weight(shape=(1, ),
                                     initializer='zeros',
                                     regularizer=None,
                                     constraint=None,
                                     name='gamma',
                                     trainable=True)


        self.built = True

    def compute_output_shape(self, input_shape):

        return (input_shape[0], input_shape[1],input_shape[2],input_shape[3])

    def call(self, input):
        input1 = input[:,:,:,:,0]
        input2 = input[:,:,:,:,1]
        input_shape = input1.get_shape().as_list()
        _, h, w, filters = input_shape

        b = Conv2D(filters // 8, 1 ,use_bias=False, kernel_initializer='he_normal')(input1)
        c = Conv2D(filters // 8, 1,use_bias=False, kernel_initializer='he_normal')(input1)
        b2 = Conv2D(filters // 8, 1 ,use_bias=False, kernel_initializer='he_normal')(input2)
        c2 = Conv2D(filters // 8, 1,use_bias=False, kernel_initializer='he_normal')(input2)
        d = Conv2D(filters, 1,use_bias=False, kernel_initializer='he_normal')(input2)

        vec_b = K1.reshape(b, (-1, h * w, filters // 8))
        vec_cT = tf.transpose(K1.reshape(c, (-1, h * w, filters // 8)), (0, 2, 1))
        bcT = K1.batch_dot(vec_b, vec_cT)
        softmax_bcT = Activation('softmax')(bcT)
        vec_b2 = K1.reshape(b2, (-1, h * w, filters // 8))
        vec_cT2 = tf.transpose(K1.reshape(c2, (-1, h * w, filters // 8)), (0, 2, 1))
        bcT2 = K1.batch_dot(vec_b2, vec_cT2)
        softmax_bcT2 = Activation('softmax')(bcT2)
        vec_d = K1.reshape(d, (-1, h * w, filters))
        bcTd = K1.batch_dot(softmax_bcT, vec_d)
        bcTd2 = K1.batch_dot(softmax_bcT2, vec_d)
        bcTd = K1.reshape(bcTd, (-1, h, w, filters))
        bcTd2 = K1.reshape(bcTd2, (-1, h, w, filters))
        out = input2 +self.gamma*bcTd +self.gamma*bcTd2
        return out


