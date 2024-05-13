# functions that handle access to models and their layers
# (later potentially depending on framework)

import warnings

from tensorflow import keras


def load_model(path):
    model = keras.models.load_model(path)
    return model

name_to_model = {"VGG16":  keras.applications.vgg16.VGG16}

def load_imagenet_pretrained_model(model_name):
    model = name_to_model[model_name]()
    return model

def get_layer_outputs_of_interest(model_layers, layer_names_of_interest):
    model_layer_outputs_of_interest = []
    model_layer_names = [l.name for l in model_layers]

    input_layer_name = model_layer_names[0]
    if input_layer_name not in layer_names_of_interest:
        layer_names_of_interest.append(input_layer_name)

    output_layer_name = model_layer_names[-1]
    if output_layer_name not in layer_names_of_interest:
        layer_names_of_interest.append(output_layer_name)

    # create a list of layer outputs which are in the layer_names_of_interest
    for l in model_layers:
        layer_name = l.name
        if layer_name in layer_names_of_interest:
            model_layer_outputs_of_interest.append(l.output)
            layer_index = layer_names_of_interest.index(layer_name)
            layer_names_of_interest.pop(layer_index)
            if len(layer_names_of_interest) == 0:
                break
    if len(layer_names_of_interest) > 0:
        warnings.warn('[' + ','.join(layer_names_of_interest) + '] not in model layers and ignored.')
    return model_layer_outputs_of_interest

def create_model_with_layer_outputs(model, layer_names_of_interest):
    model_layer_outputs_of_interest = get_layer_outputs_of_interest(model.layers, layer_names_of_interest)
    model_with_layer_outputs = keras.Model(model.inputs, model_layer_outputs_of_interest)

    return model_with_layer_outputs

