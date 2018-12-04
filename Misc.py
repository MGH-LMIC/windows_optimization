
import numpy as np

# [WL, WW]
window_settings = {
    "abdomen": [40., 400.],
    "bone": [300., 1500.],
    "brain": [50., 100.],
}

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