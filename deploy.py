
import subprocess
import shlex

from functions import *
from config import *

def check_model_exist():

def check_model_exist():


if __name__ == "__main__":

    if check_image_exist() == False:
        os.system("bash download_images.sh")

    if check_model_exist() == False:
        os.system("bash download_models.sh")

    #-----------------------------------------------------------
    # Set up configurations
    #-----------------------------------------------------------
    task = "ich"
    labels = ["ich-negative", "ich-positive"]
    model_name = "inceptionv3_ich_sigmoid_U255_nch2"
    # task = "stone"
    # labels = ["stone-negative", "stone-positive"]
    # model_name = "inceptionv3_stone_sigmoid_U255_nch2"

    # window function configurations
    window_func = "sigmoid"
    window_upbound = 255
    window_nch = 2

    list_gpus = [0]

    image_dir = "{}/{}".format(image_base_dir, task)
    model_dir = "{}/{}".format(model_base_dir, model_name)

    # set gpu
    set_gpu_config(list_gpus)
    num_labels = len(labels)

    # make result dir
    result_dir = "{}/{}".format(result_base_dir, model_name)
    if not os.path.exists(result_dir):
        command = "mkdir -p {}".format(result_dir)
        subprocess.call(shlex.split(command))

    #-----------------------------------------------------------
    # Load model and weights
    #-----------------------------------------------------------
    model_json_file = "{}/model.json".format(model_dir)
    model_weights_file = "{}/weights.h5".format(model_dir)
    model = load_model(model_json_file, model_weights_file, window_func, window_upbound)

    #----------------------------------------------
    # Test
    #----------------------------------------------
    model_func = define_model_func(model)

    image_idx = 0
    batches = 0
    dict_results = {"gt":[], "pred":[], "prob":[]}
    # shape: batch_test_images = (1,512,512,1), batch_test_labels = (1,1)

    for image_file in os.listdir(image_dir):
        if image_file.startswith("."):
            continue
        image_name = image_file.split(".")[0]
        image_file_path = "{}/{}".format(image_dir, image_file)
        # load image
        ori_image = np.load(image_file_path)
        test_image = ori_image[np.newaxis,:,:,np.newaxis]

        # deploy model
        prob, windowed_maps = model_func([test_image, 0])
        prob = prob[0,0]
        pred = 1 if prob>=0.5 else 0

        # save original image
        save_image("{}/{}.png".format(result_dir, image_name), ori_image, hist_equal=True)
        # save windowed images
        for ch_idx in range(window_nch):
            window_image = windowed_maps[0,:,:,ch_idx]
            result_file_path = "{}/{}_window_ch{}.png".format(result_dir, image_name, ch_idx+1)
            save_image(result_file_path, window_image, hist_equal=True)

        print("[{}] prediction : {}".format(image_name, labels[pred]))