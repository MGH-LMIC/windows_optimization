
import numpy as np

# configurations for dirname
image_base_dir = "images"
model_base_dir = "models"
result_base_dir = "results"

# [WL, WW]
dict_window_settings = {
    "brain": (50., 100.),
    "subdural": (50., 130.),
    "abdomen": (40., 400.),
    "bone": (300., 1500.),
    "ich_init": ["brain", "subdural"],
    "stone_init": ["bone", "abdomen"],
}
