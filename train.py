# This is the code to train model.
#
# Author: Weijie Wei
# Date: 03 / Jun / 2020
#
from __future__ import division
import argparse
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import os, os.path
import numpy as np
from config import *
from utilities import preprocess_images, preprocess_maps, preprocess_fixmaps
from models import ASD_MF


def generator(b_s, root_path, phase_gen='train', args=None):
    imgs_path = root_path + '/Images/'
    maps_path = root_path + '/FixMaps/'
    fixs_path = root_path + '/FixPts/'

    images = [imgs_path + f for f in os.listdir(imgs_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    maps = [maps_path + f for f in os.listdir(maps_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    fixs = [fixs_path + f for f in os.listdir(fixs_path) if f.endswith('.mat')]

    images.sort()
    maps.sort()
    fixs.sort()

    gaussian = np.zeros((b_s, nb_gaussian, shape_r_gt, shape_c_gt))

    counter = 0
    while True:
        Y = preprocess_maps(maps[counter:counter+b_s], shape_r_out, shape_c_out)

        # Single-side clip, valid when train on ASD dataset
        if args.ssc and phase_gen == 'train':
            Y[Y<0.1] = 0

        Y_fix = preprocess_fixmaps(fixs[counter:counter + b_s], shape_r_out, shape_c_out)
        X = preprocess_images(images[counter:counter + b_s], shape_r, shape_c)

        yield [X, gaussian], [Y, Y, Y_fix,
                              Y, Y, Y_fix,
                              Y, Y, Y_fix,
                              Y, Y, Y_fix,
                              # DOF,
                              ]
        counter = (counter + b_s) % len(images)


def lr_sch(epoch):
    # 'epoch' start from 0
    lr = float(K.get_value(model.optimizer.lr))
    if (epoch!=0 and epoch%3==0):
        lr = lr * 0.1
        print('lr: %f' % lr)

    return lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Method setting
    parser.add_argument('--ssc', action='store_true',
                        help='apply single side clip (SSC)')
    # parser.add_argument('--model_path', default='weights/weights--1.4651.pkl', type=str)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--output_path', default='weights/', type=str)
    parser.add_argument('--train_set_path', default='path/to/training/set/', type=str)
    parser.add_argument('--val_set_path', default='path/to/validation/set/', type=str)

    # Model setting
    parser.add_argument('--init_lr', default=1e-4, type=float)

    # parser.add_argument('model_path', type=pl.Path,
    #                     help='the path of the pre-trained model')
    # parser.add_argument('img_path', type=pl.Path,
    #                     help='the folder of salicon data')

    # parser.add_argument('--gpu', default='0', type=str,
    #                     help='The index of the gpu you want to use')
    # parser.add_argument('--size', default=(480, 640), type=tuple,
    #                     help='resize the input image, (640,480) is from the training data, SALICON.')
    #
    args = parser.parse_args()

    model = ASD_MF(args)
    lr_decay = LearningRateScheduler(lr_sch)

    if args.model_path is not None:
        model.load_weights(args.model_path)

    print("Training ...")
    path = args.output_path
    model.fit_generator(generator(b_s=b_s, phase_gen='train', args=args),
                        steps_per_epoch = (nb_imgs_train//b_s), epochs=epochs,#verbose=2,
                        validation_data=generator(b_s=b_s, phase_gen='val', args=args),
                        validation_steps=(nb_imgs_val//b_s),
                        callbacks=[EarlyStopping(patience=10),
                                   ModelCheckpoint(path+'.weights.{epoch:02d}-{val_loss:.4f}-{loss:.4f}.pkl', save_best_only=False),
                                   lr_decay],
                        )