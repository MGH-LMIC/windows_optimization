import numpy as np
import keras
from keras import layers
from keras import backend as K
from keras import regularizers
from keras.preprocessing import image
from keras.initializers import Constant
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
def WindowOptimizer(act_window="sigmoid", upbound_window=255, nch_window=1, **kwargs):
    '''
    :param act_window: str. sigmoid or relu
    :param upbound_window: float. a upbound value of window
    :param nch_window: int. number of channels
    :param init_windows: str or list. If list, len of list should be same with nch_window.
    :param kwargs: other parameters for convolution layer.

    :return: windows optimizer layer

    # Input shape
        Arbitrary.
    # Output shape
        Same shape as input.
    '''

    ## TODO: customizable layer name
    wc_name = 'window_conv'
    wa_name = 'window_act'

    conv_layer = WinOptConv(nch_window=nch_window, conv_layer_name=wc_name)
    act_layer = WinOptActivation(act_window=act_window, upbound_window=upbound_window, act_layer_name=wa_name)

    ## Return layer funcion
    def window_func(x):
        x = conv_layer(x)
        x = act_layer(x)
        return x

    return window_func

def WinOptConv(nch_window, conv_layer_name):
    conv_layer = Conv2D(filters=nch_window, kernel_size=(1, 1), strides=(1, 1), padding="same",
                        name=conv_layer_name,
                        kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(0.5 * 1e-5))
    return conv_layer

def WinOptActivation(act_window, upbound_window, act_layer_name):
    def upbound_relu(x):
        return K.minimum(K.maximum(x, 0), upbound_window)

    def upbound_sigmoid(x):
        return upbound_window * K.sigmoid(x)

    if act_window == "relu":
        act_layer = Activation(upbound_relu, name=act_layer_name)
    elif act_window == "sigmoid":
        act_layer = Activation(upbound_sigmoid, name=act_layer_name)
    else:
        ## Todo: make a proper exception for here
        raise Exception()

    return act_layer

# def get_initial_parameter_with_name(window_name, act_window, upbound_value):
#     ## Get window settings from dictionay
#     wl, ww = window_settings[window_name]
#     ## Set convolution layer
#     w_new, b_new = get_init_conv_params(wl, ww, act_window, upbound_value)
#     return w_new, b_new

# def load_weight_layer(layer, act_window, window_names='abdomen', upbound_value=255.0):
#     '''
#     :param layer: 1x1 conv layer to initialize
#     :param act_window: str. sigmoid or relu
#     :param window_names: str. name of predefined window setting to init
#     :param upbound_value: float. default 255.0
#     :return:
#     '''
#
#     ## Get window settings from dictionay
#     wl, ww = window_settings[window_names]
#
#     w_new, b_new = get_init_conv_params(wl, ww, act_window, upbound_value)
#     w_conv_ori, b_conv_ori = layer.get_weights()
#     w_conv_new = np.zeros_like(w_conv_ori)
#     w_conv_new[0, 0, 0, :] = w_new * np.ones(w_conv_ori.shape[-1], dtype=w_conv_ori.dtype)
#     b_conv_new = b_new * np.ones(b_conv_ori.shape, dtype=b_conv_ori.dtype)
#     layer.set_weights([w_conv_new, b_conv_new])
#
#     return layer

def initialize_window_setting(model, act_window="sigmoid", init_windows="abdomen", conv_layer_name="window_conv"):

    # get all layer names
    layer_names = [layer.name for layer in model.layers]

    # multi-channel window settings
    if init_windows in ["stone", "ich"]:
        n_window_settings = len(dict_window_settings[init_windows])
        w_conv_ori, b_conv_ori = model.layers[layer_names.index(conv_layer_name)].get_weights()
        n_windows = w_conv_ori.shape[-1]
        w_conv_new = np.zeros((1,1,1,n_windows), dtype=np.float32)
        b_conv_new = np.zeros(n_windows, dtype=np.float32)
        for idx, window_setting in enumerate(dict_window_settings[init_windows]):
            wl, ww = window_setting
            if act_window == "relu":
                w_new, b_new = get_init_conv_params_relu(wl, ww)
            elif act_window == "sigmoid":
                w_new, b_new = get_init_conv_params_sigmoid(wl, ww)

            w_conv_new[0,0,0,range(idx, n_windows, n_window_settings)] = w_new
            b_conv_new[range(idx, n_windows, n_window_settings)] = b_new

        model.layers[layer_names.index(conv_layer_name)].set_weights([w_conv_new, b_conv_new])
    # single-channel window setting
    else:
        wl, ww = dict_window_settings[init_windows]
        if act_window == "relu":
            w_new, b_new = get_init_conv_params_relu(wl, ww)
        elif act_window == "sigmoid":
            w_new, b_new = get_init_conv_params_sigmoid(wl, ww)

        w_conv_ori, b_conv_ori = model.layers[layer_names.index(conv_layer_name)].get_weights()
        w_conv_new = np.zeros_like(w_conv_ori)
        w_conv_new[0,0,0,:] = w_new * np.ones(w_conv_ori.shape[-1], dtype=w_conv_ori.dtype)
        b_conv_new = b_new * np.ones(b_conv_ori.shape, dtype=b_conv_ori.dtype)

        model.layers[layer_names.index(conv_layer_name)].set_weights([w_conv_new, b_conv_new])

    # print("window_func={}, window_name={} [{}]".format(window_func, window_name, dict_window_settings[window_name]))
    # print(model.layers[layer_names.index(conv_layer_name)].get_weights())

    return model

# def load_pretrained_weights(model, conv_layer_name='window_conv', classifier='ICH'):
#     ## TODO : Load pretrained weights of ICH and Stone classifier.
#     pass


if __name__ == "__main__":

    ## WSO configurations
    nch_window = 2
    act_window = "sigmoid"
    upbound_window = 255
    init_windows = "ich"


    # x = load_example_dicom() # They should be 2d-HU values matrix
    input_shape = (512, 512, 1)
    input_tensor = keras.layers.Input(shape=input_shape, name="input")

    #### NOTE
    ## Define a window setting optimization layer
    x = WindowOptimizer(nch_window=nch_window, act_window=act_window, upbound_window=upbound_window)(input_tensor)

    ## ... add some your layer here
    x = Conv2D(32, (3, 3), activation=None, padding="same", name="conv1")(x)
    x = Activation("relu", name="conv1_relu")(x)
    x = MaxPooling2D((7, 7), strides=(3, 3), name="pool1")(x)
    x = Conv2D(256, (3, 3), activation=None, padding="same", name="conv2")(x)
    x = Activation("relu", name="conv2_relu")(x)
    x = MaxPooling2D((7, 7), strides=(3, 3), name="pool2")(x)
    x = GlobalAveragePooling2D(name="gap")(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_tensor, outputs=outputs, name="main_model")

    #### NOTE
    ## Initialize parameters of window setting opt module
    model = initialize_window_setting(model, act_window=act_window, init_windows=init_windows)

    optimizer = SGD(lr=0.0001, decay=0, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])
    model.summary()

    ## Double check initialized parameters for WSO
    names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()

    for name, weight in zip(names, weights):
        if "window_conv" in name:
            if "kernel:0" in name:
                ws = weight
            if "bias:0" in name:
                bs = weight
    print("window optimization modeul set up (initialized with {} settings)".format(init_windows))
    print("(WL, WW)={}".format(dict_window_settings[init_windows]))
    print("w={} b={}".format(ws[0, 0, 0, :], bs))

