
import os
import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import model_from_json

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
