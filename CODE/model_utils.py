# model_utils.py
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

def load_model_file(name):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f'Model file not found: {path}')
    model = load_model(path)
    return model

def prepare_image(filepath, target_size=(224,224)):
    """
    Return image as float32 array in range [0,255].
    Do NOT divide by 255 here — call preprocess_input() in app.py to
    match EfficientNet preprocessing.
    """
    img = image.load_img(filepath, target_size=target_size)
    x = image.img_to_array(img).astype('float32')
    x = np.expand_dims(x, axis=0)
    return x
