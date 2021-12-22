import numpy as np
from tensorflow import keras
from keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

class_species = ['EDIBLE',
                 'POISONOUS_AMANITA_MUSCARIA',
                 'POISONOUS_AMANITA_PHALLOIDES',
                 'POISONOUS_AMANITA_VERNA',
                 'POISONOUS_AMANITA_VIROSA',
                 'POISONOUS_CLITOCYBE_DEALBATA',
                 'POISONOUS_CORTINARIUS_RUBELLUS',
                 'POISONOUS_GALERINA_MARGINATA',
                 'POISONOUS_GYROMITRA_ESCULENTA',
                 'POISONOUS_PLEUROCYBELLA_PORRIGENS',
                 'POISONOUS_PODOSTROMA_CORNUDAMAE']

class_species = [i.replace('POISONOUS_', '').replace(
    '_', ' ').lower().title() for i in class_species]


def load_model(path_model):
    model = keras.models.load_model(path_model)
    return model


def load_image(path_image, target_size=(227, 227, 3)):
    img = image.load_img(path_image, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def predict(model, path_image, classification='base_2_type'):
    input_img = load_image(path_image)
    result_predict = model.predict(input_img)[0]
    if classification == 'base_2_type':
        result_max = 1 if result_predict[0] > 0.5 else 0
        if result_max:
            probability = result_predict[0]
            class_name = 'Poisonous'
        else:
            probability = 1 - result_predict[0]
            class_name = 'Edible'
    else:
        result_max = np.argmax(result_predict, axis=-1)
        probability = result_predict[result_max]
        class_name = class_species[result_max]
    return result_max, class_name, str(round(probability * 100, 2)) + ' %'
