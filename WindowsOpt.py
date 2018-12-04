import numpy as np
import keras
from keras import layers
from keras import backend as K
from keras import regularizers
from keras.preprocessing import image
from keras.layers import Input, Dropout, Lambda, MaxPooling1D, MaxPooling2D, Dense, Activation, Concatenate, GlobalAveragePooling2D, GlobalAveragePooling1D, AveragePooling2D, UpSampling2D, ZeroPadding2D, Softmax, Layer
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, Callback
from keras.models import Model, model_from_json
from keras.applications.inception_v3 import InceptionV3

from Misc import *

#-------------------------------------
# Window Optimization layer : simple function
#-------------------------------------
def WindowOptimizer(inputs, act_window="sigmoid", upbound_window=255, nch_window=1, init_windows='abdomen', **kwargs):
    '''
    :param inputs: input tensor
    :param act_window: str. sigmoid or relu
    :param upbound_window: float. a upbound value of window
    :param nch_window: int. number of channels
    :param init_windows: str or list. If list, len of list should be same with nch_window.
    :param kwargs: other parameters for convolution layer.

    :return: output tensor

    # Input shape
        Arbitrary.
    # Output shape
        Same shape as input.
    '''

    ## Check parameter integrity.
    if nch_window == 1:
        assert(type(init_windows) == str)
    else:
        assert(type(init_windows) == list and len(init_windows) == nch_window)

    ## TODO: customize layer name
    wc_name = 'window_conv'
    wa_name = 'window_act'

    ## Set convolution layer
    conv_layer = Conv2D(filters=nch_window, kernel_size=(1, 1), strides=(1, 1), padding="same", name=wc_name, **kwargs)

    ## Set activation layer
    act_layer = WinOptActivation(act_window=act_window, upbound_window=upbound_window, name=wa_name)

    ## Initialize convolution layer
    initialize_layer(conv_layer, act_window=act_window, window_names=init_windows, upbound_value=upbound_window)

    ## Return layer funcion
    def window_func(x):
        x = conv_layer(x)
        x = act_layer(x)
        return x

    return window_func


def WinOptActivation(act_window, upbound_window, name):
    def upbound_relu(x):
        return K.minimum(K.maximum(x,0),upbound_window)

    def upbound_sigmoid(x):
        return upbound_window*K.sigmoid(x)

    if act_window == "relu":
        act_layer = Activation(upbound_relu, name=name)
    elif act_window == "sigmoid":
        act_layer = Activation(upbound_sigmoid, name=name)
    else:
        ## Todo: make a proper exception for here
        raise Exception()

    return act_layer


def initialize_layer(layer, act_window, window_names='abdomen', upbound_value=255.0):
    '''
    :param layer: 1x1 conv layer to initialize
    :param act_window: str. sigmoid or relu
    :param window_names: str. name of predefined window setting to init
    :param upbound_value: float. default 255.0
    :return:
    '''

    ## TODO : Make function for when conv layer have mutiple channels. Now it's only support for one channel.

    ## Get window settings from dictionay
    wl, ww = window_settings[window_names]

    if act_window == 'sigmoid':
        w_new, b_new = get_init_conv_params_sigmoid(wl, ww, upbound_value=upbound_value)
    elif act_window == 'relu':
        w_new, b_new = get_init_conv_params_relu(wl, ww, upbound_value=upbound_value)
    else:
        ## TODO : make a proper exception
        raise Exception()

    w_conv_ori, b_conv_ori = layer.get_weights()
    w_conv_new = np.zeros_like(w_conv_ori)
    w_conv_new[0, 0, 0, :] = w_new * np.ones(w_conv_ori.shape[-1], dtype=w_conv_ori.dtype)
    b_conv_new = b_new * np.ones(b_conv_ori.shape, dtype=b_conv_ori.dtype)
    layer.set_weights([w_conv_new, b_conv_new])

    return layer