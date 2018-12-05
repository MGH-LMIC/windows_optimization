
import os
import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import model_from_json

# [WL, WW]
dict_window_settings = {
    "brain": (50., 100.),
    "subdural": (50., 130.),
    "abdomen": (40., 400.),
    "bone": (300., 1500.),
    "ich_init": ["brain", "subdural"],
    "stone_init": ["bone", "abdomen"],
}

def set_gpu_config(gpu_ids):
    # set gpu_id
    str_gpu_id = ",".join([str(item) for item in gpu_ids])
    os.environ["CUDA_VISIBLE_DEVICES"] = str_gpu_id

    # make TF not to allocate all of the memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)

def load_model(model_json_file, model_weights_file, window_func, window_upbound):

    def upbound_relu(x, window_upbound=window_upbound):
        return K.minimum(K.maximum(x,0),window_upbound)

    def upbound_sigmoid(x, window_upbound=window_upbound):
        return window_upbound*K.sigmoid(x)

    # load model
    jf = open(model_json_file, "r")
    model_json = jf.read()
    jf.close()
    if window_func == "relu":
        model = model_from_json(model_json, custom_objects={"upbound_relu":upbound_relu})
    elif window_func == "sigmoid":
        model = model_from_json(model_json, custom_objects={"upbound_sigmoid":upbound_sigmoid})
    else:
        raise ValueError()

    # load weights
    model.load_weights(model_weights_file)

    return model

def define_model_func(model):
    predict = model.get_layer("predict").output
    windowed_maps = model.get_layer("window_act").output
    model_func = K.function([model.input, K.learning_phase()], [predict, windowed_maps])
    return model_func

def histogram_equalization(img):
    """
        Convert min ~ max range to 0 ~ 255 range
    """

    img = img.astype(np.float64)

    min_val = np.min(img)
    max_val = np.max(img)
    img = img - min_val
    img = img/(max_val-min_val+1e-6)

    img = img*255.0
    img = img.astype(np.uint8)

    return img

def save_image(result_file_path, image, hist_equal=False):
    if hist_equal:
        image = histogram_equalization(image)

    # save original image
    cv2.imwrite(result_file_path, image)


def get_init_conv_params(wl, ww, act_window, upbound_value):
    if act_window == 'sigmoid':
        w_new, b_new = get_init_conv_params_sigmoid(wl, ww, upbound_value=upbound_value)
    elif act_window == 'relu':
        w_new, b_new = get_init_conv_params_relu(wl, ww, upbound_value=upbound_value)
    else:
        ## TODO : make a proper exception
        raise Exception()
    return w_new, b_new

def get_init_conv_params_relu(wl, ww, upbound_value=255.):
    w = upbound_value / ww
    b = -1. * upbound_value * (wl - ww / 2.) / ww
    return (w, b)

def get_init_conv_params_sigmoid(wl, ww, smooth=1., upbound_value=255.):
    w = 2./ww * np.log(upbound_value/smooth - 1.)
    b = -2.*wl/ww * np.log(upbound_value/smooth - 1.)
    return (w, b)

def get_window_settings_relu(w, b, upbound_value=255.):
    wl = upbound_value/(2.*w) - b/w
    ww = upbound_value / w
    return (wl, ww)

def get_window_settings_sigmoid(w, b, smooth=1., upbound_value=255.):
    wl = b/w
    ww = 2./w * np.log(upbound_value/smooth - 1.)
    return (wl, ww)

def get_pretrained_model_from_internet():
    ## TODO : A Pretrained Inception-V3 model with ICH, StoneAI
    pass
