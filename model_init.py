from my_utils import *

def load_model_from_path(weight_path):
    # load the model
    try:
        model = load_model(weight_path, get_device())
        print("model loading completed")
        return model
    except Exception as e:
        print("Exception in model loading : {} ".format(e))


def get_model_device():
    return get_device()