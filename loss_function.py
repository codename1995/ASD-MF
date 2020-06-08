import keras.backend as K
from config import *

# KL-Divergence Loss
def KL(y_true, y_pred):
    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)),
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    y_pred /= max_y_pred

    sum_y_true = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_true, axis=2), axis=2)),
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    sum_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_pred, axis=2), axis=2)),
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())

    return 10 * K.sum(K.sum(y_true * K.log((y_true / (y_pred + K.epsilon())) + K.epsilon()), axis=-1), axis=-1)


# Correlation Coefficient Loss
def CC(y_true, y_pred):
    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)),
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    y_pred /= max_y_pred

    sum_y_true = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_true, axis=2), axis=2)),
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    sum_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_pred, axis=2), axis=2)),
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)

    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())

    N = shape_r_out * shape_c_out
    sum_prod = K.sum(K.sum(y_true * y_pred, axis=2), axis=2)
    sum_x = K.sum(K.sum(y_true, axis=2), axis=2)
    sum_y = K.sum(K.sum(y_pred, axis=2), axis=2)
    sum_x_square = K.sum(K.sum(K.square(y_true), axis=2), axis=2)
    sum_y_square = K.sum(K.sum(K.square(y_pred), axis=2), axis=2)

    num = sum_prod - ((sum_x * sum_y) / N)
    den = K.sqrt((sum_x_square - K.square(sum_x) / N) * (sum_y_square - K.square(sum_y) / N))

    return -2 * num / den


# Normalized Scanpath Saliency Loss
def NSS(y_true, y_pred):
    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)),
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    y_pred /= max_y_pred
    y_pred_flatten = K.batch_flatten(y_pred)

    y_mean = K.mean(y_pred_flatten, axis=-1)
    y_mean = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.expand_dims(y_mean)),
                                                               shape_r_out, axis=-1)), shape_c_out, axis=-1)

    y_std = K.std(y_pred_flatten, axis=-1)
    y_std = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.expand_dims(y_std)),
                                                              shape_r_out, axis=-1)), shape_c_out, axis=-1)

    y_pred = (y_pred - y_mean) / (y_std + K.epsilon())

    return -(K.sum(K.sum(y_true * y_pred, axis=2), axis=2) / K.sum(K.sum(y_true, axis=2), axis=2))

# KL-Divergence Loss
def KL_1_3(y_true, y_pred):
    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)),
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    y_pred /= max_y_pred

    sum_y_true = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_true, axis=2), axis=2)),
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    sum_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_pred, axis=2), axis=2)),
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())

    return 10 * K.sum(K.sum(y_true * K.log((y_true / (y_pred + K.epsilon())) + K.epsilon()), axis=-1), axis=-1)/3


# Correlation Coefficient Loss
def CC_1_3(y_true, y_pred):
    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)),
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    y_pred /= max_y_pred

    sum_y_true = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_true, axis=2), axis=2)),
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    sum_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_pred, axis=2), axis=2)),
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)

    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())

    N = shape_r_out * shape_c_out
    sum_prod = K.sum(K.sum(y_true * y_pred, axis=2), axis=2)
    sum_x = K.sum(K.sum(y_true, axis=2), axis=2)
    sum_y = K.sum(K.sum(y_pred, axis=2), axis=2)
    sum_x_square = K.sum(K.sum(K.square(y_true), axis=2), axis=2)
    sum_y_square = K.sum(K.sum(K.square(y_pred), axis=2), axis=2)

    num = sum_prod - ((sum_x * sum_y) / N)
    den = K.sqrt((sum_x_square - K.square(sum_x) / N) * (sum_y_square - K.square(sum_y) / N))

    return -2 * num / den/3


# Normalized Scanpath Saliency Loss
def NSS_1_3(y_true, y_pred):
    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)),
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    y_pred /= max_y_pred
    y_pred_flatten = K.batch_flatten(y_pred)

    y_mean = K.mean(y_pred_flatten, axis=-1)
    y_mean = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.expand_dims(y_mean)),
                                                               shape_r_out, axis=-1)), shape_c_out, axis=-1)

    y_std = K.std(y_pred_flatten, axis=-1)
    y_std = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.expand_dims(y_std)),
                                                              shape_r_out, axis=-1)), shape_c_out, axis=-1)

    y_pred = (y_pred - y_mean) / (y_std + K.epsilon())

    return -(K.sum(K.sum(y_true * y_pred, axis=2), axis=2) / K.sum(K.sum(y_true, axis=2), axis=2))/3

# DRE Loss
def DRE(y_true, y_pred):
    shape_r_out = y_pred._keras_shape[2]
    shape_c_out = y_pred._keras_shape[3]

    # Min-Max Normalization
    min_y_pred = K.repeat_elements(K.expand_dims(
                 K.repeat_elements(K.expand_dims(
                     K.min(K.min(y_pred, axis=3), axis=2)
                     , axis=2), shape_r_out, axis=2)
                     , axis=3), shape_c_out, axis=3)
    y_pred = y_pred - min_y_pred
    max_y_pred = K.repeat_elements(K.expand_dims(
                 K.repeat_elements(K.expand_dims(
                     K.max(K.max(y_pred, axis=3), axis=2)
                     , axis=2), shape_r_out, axis=2)
                     , axis=3), shape_c_out, axis=3)
    y_pred /= (max_y_pred + eps)

    return K.sum(K.sum(y_true * K.abs(y_pred-y_true), axis=3), axis=2) / K.sum(K.sum(y_true, axis=3), axis=2)
    # return 2*K.sum(K.sum(y_true * K.square(y_pred-y_true), axis=2), axis=1) / K.sum(K.sum(y_true, axis=2), axis=1)