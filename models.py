from __future__ import division
from keras.layers import Input, Concatenate, Conv2DTranspose, Cropping2D, Conv2D, MaxPooling2D, Lambda
from keras.layers.merge import concatenate
from keras.layers.convolutional import AtrousConvolution2D
from keras.models import Model
from keras.optimizers import RMSprop
import keras.backend as K
import theano.tensor as T
import os
from gaussian_prior import LearningPrior
from loss_function import KL, CC, NSS, KL_1_3, CC_1_3, NSS_1_3, DRE
from config import *

def upsampling(x):
    return T.nnet.abstract_conv.bilinear_upsampling(input=x, ratio=upsampling_factor, num_input_channels=1, batch_size=b_s)

def upsampling_shape(s):
    return s[:2] + (s[2] * upsampling_factor, s[3] * upsampling_factor)

def DCN(input_tensor=None):
    input_shape = (3, None, None)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # conv_1
    x = Conv2D(64, [3, 3], activation='relu', padding='same', name='block1_conv1', data_format='channels_first')(img_input)
    x = Conv2D(64, [3, 3], activation='relu', padding='same', name='block1_conv2', data_format='channels_first')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format='channels_first')(x)

    # conv_2
    x = Conv2D(128, [3, 3], activation='relu', padding='same', name='block2_conv1', data_format='channels_first')(x)
    x = Conv2D(128, [3, 3], activation='relu', padding='same', name='block2_conv2', data_format='channels_first')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format='channels_first')(x)

    # conv_3
    x = Conv2D(256, [3, 3], activation='relu', padding='same', name='block3_conv1', data_format='channels_first')(x)
    x = Conv2D(256, [3, 3], activation='relu', padding='same', name='block3_conv2', data_format='channels_first')(x)
    x = Conv2D(256, [3, 3], activation='relu', padding='same', name='block3_conv3', data_format='channels_first')(x)
    x_conv3_3 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', padding='same', data_format='channels_first')(x)


    # conv_4
    x = Conv2D(512, [3, 3], activation='relu', padding='same', name='block4_conv1', data_format='channels_first')(x)
    x = Conv2D(512, [3, 3], activation='relu', padding='same', name='block4_conv2', data_format='channels_first')(x)
    x = Conv2D(512, [3, 3], activation='relu', padding='same', name='block4_conv3', data_format='channels_first')(x)
    x_conv4_3 = x
    x = MaxPooling2D((2, 2), strides=(1, 1), name='block4_pool', padding='same', data_format='channels_first')(x)

    # conv_5
    x = Conv2D(512, [3, 3], activation='relu', padding='same', name='block5_conv1', dilation_rate=(2, 2), data_format='channels_first')(x)
    x = Conv2D(512, [3, 3], activation='relu', padding='same', name='block5_conv2', dilation_rate=(2, 2), data_format='channels_first')(x)
    x = Conv2D(512, [3, 3], activation='relu', padding='same', name='block5_conv3', dilation_rate=(2, 2), data_format='channels_first')(x)

    # Create model
    model = Model(img_input, [x_conv3_3, x_conv4_3, x])

    # Load imagenet pretrained weights
    vgg16_weights = 'baseline_weights/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
    if os.path.exists(vgg16_weights):
        model.load_weights(vgg16_weights)

    return model


def Decoder(input_tensor=None):

    conv_3_3 = input_tensor[0]
    conv_4_3 = input_tensor[1]
    conv_5_3 = input_tensor[2]

    h_0 = conv_5_3._keras_shape[2] * 8
    w_0 = conv_5_3._keras_shape[3] * 8

    # UP NETWORK 1 ###########################################################################
    deconv5_1 = Conv2DTranspose(256, kernel_size = [4, 4], strides=2,activation='relu',name='deconv5_2')(conv_5_3)
    deconv5_2 = Conv2DTranspose(128, kernel_size = [4, 4], strides=2,activation='relu',name='deconv5_3')(deconv5_1)
    deconv5_3 = Conv2DTranspose(64, kernel_size = [4, 4], strides=2,activation='relu',name='deconv5_4')(deconv5_2)
    h_5 = deconv5_3._keras_shape[2]
    w_5 = deconv5_3._keras_shape[3]
    dh5 = (int((h_5-h_0)/2))
    dw5 = (int((w_5-w_0)/2))
    deconv5_C = Cropping2D((dh5,dw5), data_format='channels_first')(deconv5_3)

    # UP NETWORK 2 ###########################################################################
    deconv4_1 = Conv2DTranspose(256, kernel_size = [4, 4], strides=2,activation='relu',name='deconv4_1')(conv_4_3)
    deconv4_2 = Conv2DTranspose(128, kernel_size = [4, 4], strides=2,activation='relu',name='deconv4_2')(deconv4_1)
    deconv4_3 = Conv2DTranspose(64, kernel_size = [4, 4], strides=2,activation='relu',name='deconv4_3')(deconv4_2)
    h_4 = deconv4_3._keras_shape[2]
    w_4 = deconv4_3._keras_shape[3]
    dh4 = (int((h_4-h_0)/2))
    dw4 = (int((w_4-w_0)/2))
    deconv4_C = Cropping2D((dh4,dw4), data_format='channels_first')(deconv4_3)

    # UP NETWORK 3 ###########################################################################
    deconv3_1 = Conv2DTranspose(128, kernel_size = [4, 4], strides=2,activation='relu',name='deconv3_1')(conv_3_3)
    deconv3_2 = Conv2DTranspose(64, kernel_size = [4, 4], strides=2,activation='relu',name='deconv3_2')(deconv3_1)
    h_3 = deconv3_2._keras_shape[2]
    w_3 = deconv3_2._keras_shape[3]
    dh3 = (int((h_3-h_0)/2))
    dw3 = (int((w_3-w_0)/2))
    deconv3_C = Cropping2D((dh3,dw3), data_format='channels_first')(deconv3_2)

    return [deconv3_C, deconv4_C, deconv5_C]

def gaussian_prior_match(tensor, fdm):
    # Learned Prior (1)
    priors1 = LearningPrior(64, nb_gaussian=nb_gaussian)(fdm[1])
    concateneted = concatenate([tensor, priors1], axis=1)
    learned_priors1 = AtrousConvolution2D(64, [5, 5], padding='same', activation='relu',
                                          atrous_rate=(4, 4))(concateneted)

    # Learned Prior (2)
    priors2 = LearningPrior(64, nb_gaussian=nb_gaussian)(fdm[1])
    concateneted = concatenate([learned_priors1, priors2], axis=1)
    learned_priors2 = AtrousConvolution2D(64, [5, 5], padding='same', activation='relu',
                                          atrous_rate=(4, 4))(concateneted)
    return learned_priors2

def SP_MODEL(x):
    # Encoder
    E = DCN(input_tensor=x[0])

    # Decoder
    D = Decoder(E.output)

    # Learned priors
    att3p = gaussian_prior_match(D[0], x)
    att4p = gaussian_prior_match(D[1], x)
    att5p = gaussian_prior_match(D[2], x)

    att3 = Conv2D(1, (1, 1), padding='same', activation='relu', name='attention3')(att3p)
    att4 = Conv2D(1, (1, 1), padding='same', activation='relu', name='attention4')(att4p)
    att5 = Conv2D(1, (1, 1), padding='same', activation='relu', name='attention5')(att5p)

    concated_att = Concatenate(axis=1)([att3,att4,att5])
    att = Conv2D(1, (1, 1), padding='same', activation='relu', name='fuse_layer')(concated_att)

    att3_up = Lambda(upsampling, upsampling_shape)(att3)
    att4_up = Lambda(upsampling, upsampling_shape)(att4)
    att5_up = Lambda(upsampling, upsampling_shape)(att5)
    att_up = Lambda(upsampling, upsampling_shape)(att)

    res = [att3_up, att3_up, att3_up,
           att4_up, att4_up, att4_up,
           att5_up, att5_up, att5_up,
           att_up, att_up, att_up,
    ]

    return res

def ASD_MF(args):
    x = Input((3, shape_r, shape_c))
    x_maps = Input((nb_gaussian, shape_r_gt, shape_c_gt))

    m = Model(inputs=[x, x_maps], outputs=SP_MODEL([x, x_maps]))

    print("Compiling ...")
    m.compile(RMSprop(lr=args.init_lr),
              loss=[KL_1_3, CC_1_3, NSS_1_3,
                    KL_1_3, CC_1_3, NSS_1_3,
                    KL_1_3, CC_1_3, NSS_1_3,
                    KL, CC, NSS,
                    # DRE,
                    ],
              )

    return m
