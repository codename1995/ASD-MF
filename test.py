# This is the code to train model.
#
# Author: Weijie Wei
# Date: 03 / Jun / 2020
#
from __future__ import division
import argparse
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import os, os.path, cv2
import numpy as np
from config import *
from utilities import preprocess_images, postprocess_predictions
from models import ASD_MF
import time

def generator_test(b_s, imgs_test_path, large_scale_dataset=False, group=0, th=500):
    images = [imgs_test_path + f for f in os.listdir(imgs_test_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()

    counter = 0
    gaussian = np.zeros((b_s, nb_gaussian, shape_r_gt, shape_c_gt))
    if large_scale_dataset:
        assert group!=0
        start = (group-1)*th
        end_ = min(group*th, len(images))
        images = images[start:end_]

        images_extend = images + images
        while True:
            yield [preprocess_images(images_extend[counter:counter + b_s], shape_r, shape_c),gaussian]
            counter = (counter + b_s) % len(images)

    else:
        images_extend = images + images
        while True:
            yield [preprocess_images(images_extend[counter:counter + b_s], shape_r, shape_c), gaussian]
            counter = (counter + b_s) % len(images)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Method setting
    parser.add_argument('--model_path', default='./weights/weights--1.4651.pkl', type=str)
    parser.add_argument('--images_path', default='./images/', type=str)
    parser.add_argument('--results_path', default='./results/', type=str)

    # Model setting
    parser.add_argument('--init_lr', default=1e-4, type=float)

    args = parser.parse_args()


    model = ASD_MF(args)

    model.load_weights(args.model_path) # Neurocomputing version MIT1003


    start_time = time.time()
    model.load_weights(args.model_path)
    print("load weights : %s"%(args.model_path))

    output_folder = args.results_path
    if os.path.isdir(output_folder) is False:
        os.makedirs(output_folder)

    images_path = args.images_path
    print("Predict saliency maps for " + images_path + " at " + output_folder)

    file_names = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    file_names.sort()

    nb_imgs_test = len(file_names)
    print(nb_imgs_test)

    th = 500
    if nb_imgs_test <= th:
        predictions0 = model.predict_generator(generator_test(b_s=5, imgs_test_path=images_path),
                                               nb_imgs_test/5)
        predictions0 = predictions0[-1]
        print(len(predictions0))
        elapsed_time2 = time.time() - start_time
        print("total model testing time: ", elapsed_time2)

        for pred, name in zip(predictions0, file_names):
            original_image = cv2.imread(images_path + name, 0)
            name = name[:-4] + '.png'
            res = postprocess_predictions(pred, original_image.shape[0], original_image.shape[1])
            cv2.imwrite(output_folder + '%s' % name, res.astype(int))
        elapsed_time = time.time() - start_time
        print("total time: ", elapsed_time)
    else:
        # split large dataset before loading
        nb_groups = nb_imgs_test // th
        print('Total group: ', nb_groups)

        for j in range(nb_groups):
            group = j + 1
            nb_imgs_test_in_this_group = th if group < nb_groups else (nb_imgs_test - (group - 1) * th)
            predictions0 = model.predict_generator(
                generator_test(b_s=1, imgs_test_path=images_path, large_scale_dataset=True, group=group),
                nb_imgs_test_in_this_group)
            predictions0 = predictions0[-1]

            start = (group - 1) * th
            end_ = min(group * th, nb_imgs_test)
            images_filename_in_this_group = file_names[start:end_]
            for pred, name in zip(predictions0, images_filename_in_this_group):
                original_image = cv2.imread(images_path + name, 0)
                name = name[:-4] + '.png'
                res = postprocess_predictions(pred, original_image.shape[0],
                                                         original_image.shape[
                                                             1])
                cv2.imwrite(output_folder + '%s' % name, res.astype(int))
            print("%d / %d" % (group, nb_groups))
        elapsed_time2 = time.time() - start_time
        print("total model testing time: ", elapsed_time2)

