# This code comes from Marcella Cornia in her TIP18 paper:
# Predicting Human Eye Fixations via an LSTM-based Saliency Attentive Model
# https://github.com/marcellacornia/sam
# We just adjusted it to Keras 2.
from __future__ import division
import keras.backend as K
import theano.tensor as T
from keras.layers import Layer, InputSpec
from keras import regularizers, constraints, activations
from keras import initializers
import theano
import numpy as np
floatX = theano.config.floatX


class LearningPrior(Layer):
    def __init__(self, units,
                 nb_gaussian,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='RandomUniform',#U(-0.05,0.05)
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        self.units = units
        self.nb_gaussian = nb_gaussian
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        # self.input_spec = InputSpec(min_ndim=2)
        self.input_spec = InputSpec(min_ndim=4)
        self.supports_masking = False
        super(LearningPrior, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel_shape = (self.nb_gaussian*4, )
        self.kernel=self.add_weight(
            name='{}_W'.format(self.name),
            shape=self.kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        init_weights = self.get_weights()
        mu = init_weights[0][0:self.nb_gaussian*2]
        mu = self.changeNormalDistribution(mu,0.15,0.85)
        sigma = init_weights[0][self.nb_gaussian*2:self.nb_gaussian*2*2]
        sigma = self.changeNormalDistribution(sigma,0.05,0.3)
        init_weights = [np.concatenate((mu[0],sigma[0]),axis=0),]
        self.set_weights(init_weights)
        self.trainable_weights = [self.kernel]

        self.built = True
        super(LearningPrior, self).build(input_shape)

    def get_output_shape_for(self, input_shape):
        self.b_s = input_shape[0]
        self.height = input_shape[2]
        self.width = input_shape[3]

        return [self.b_s, self.nb_gaussian, self.height, self.width]

    def call(self, x, mask=None):
        # 1.取 模型参数=二维高斯分布的四个参数
        mu_x = self.kernel[:self.nb_gaussian]
        mu_y = self.kernel[self.nb_gaussian:self.nb_gaussian*2]
        sigma_x = self.kernel[self.nb_gaussian*2:self.nb_gaussian*3]
        sigma_y = self.kernel[self.nb_gaussian*3:]

        # 2.取 超参数/固定参数（batch_size, H, W)
        self.b_s = x.shape[0]
        self.height = x.shape[2]
        self.width = x.shape[3]

        # 3. 长宽比相关的一些固定参数
        e = self.height / self.width
        e1 = (1 - e) / 2
        e2 = e1 + e

        # 4. clip
        mu_x = K.clip(mu_x, 0.25, 0.75)
        mu_y = K.clip(mu_y, 0.35, 0.65)

        sigma_x = K.clip(sigma_x, 0.1, 0.9)
        sigma_y = K.clip(sigma_y, 0.2, 0.8)

        # 5. 展开x方向和y方向的张量
        x_t = T.dot(T.ones((self.height, 1)), self._linspace(0, 1.0, self.width).dimshuffle('x', 0))
        y_t = T.dot(self._linspace(e1, e2, self.height).dimshuffle(0, 'x'), T.ones((1, self.width)))

        x_t = K.repeat_elements(K.expand_dims(x_t, axis=-1), self.nb_gaussian, axis=-1)
        y_t = K.repeat_elements(K.expand_dims(y_t, axis=-1), self.nb_gaussian, axis=-1)

        gaussian = 1 / (2 * np.pi * sigma_x * sigma_y + K.epsilon()) * \
                   T.exp(-((x_t - mu_x) ** 2 / (2 * sigma_x ** 2 + K.epsilon()) +
                           (y_t - mu_y) ** 2 / (2 * sigma_y ** 2 + K.epsilon())))

        gaussian = K.permute_dimensions(gaussian, (2, 0, 1))
        max_gauss = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(gaussian, axis=1), axis=1)), self.height, axis=-1)), self.width, axis=-1)
        gaussian = gaussian / max_gauss

        output = K.repeat_elements(K.expand_dims(gaussian, axis=0), self.b_s, axis=0)

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        return tuple([input_shape[0], self.nb_gaussian, input_shape[2], input_shape[3]])

    @staticmethod
    def _linspace(start, stop, num):
        # produces results identical to:
        # np.linspace(start, stop, num)
        start = T.cast(start, floatX)
        stop = T.cast(stop, floatX)
        num = T.cast(num, floatX)
        step = (stop - start) / (num - 1)
        return T.arange(num, dtype=floatX) * step + start

    def get_config(self):
        config = {'nb_gaussian': self.nb_gaussian,
                  'kernel_initializer': self.kernel_initializer,
                  'kernel_regularizer': self.kernel_regularizer if self.kernel_regularizer else None,
                  'activity_regularizer': self.activity_regularizer if self.activity_regularizer else None,
                  'kernel_constraint': self.kernel_constraint if self.kernel_constraint else None,
                  }
        base_config = super(LearningPrior, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def changeNormalDistribution(self,x,minValue,maxValue):
        if np.size(x) != 0:
            realMaxValue = np.max(x)
            realMinValue = np.min(x)
            x = (x-realMinValue)/(realMaxValue-realMinValue)
            x = np.squeeze(x)
            x_back = x*(maxValue-minValue)+minValue
            return (x_back,)