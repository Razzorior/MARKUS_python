import tensorflow as tf
from sailency_methods.integrated_gradients import integrated_gradients
import time
import tensorflow.keras as keras
import json
import numpy as np
import pickle
import os
import custom_objects
from topomaps.src.neuron_activations import get_NWPs
from helper_functions import convert_ndarray
from enum import IntEnum
from umap import UMAP


class UMAPData(IntEnum):
    output = 0
    input = 1
    input_and_output = 2
    activations = 3
    subset_signals_output = 4
    subset_signals_input = 5
    subset_signals_input_and_output = 6
    subset_activations = 7

class PythonState:
    def __init__(self):
        self.model = None
        self.model_name = None
        folder_path = "saved_models/"
        self.model_list = [file.split('.')[0] for file in os.listdir(folder_path) if file.endswith('.keras')]
        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None
        self.input = None
        self.time_until_timeout_secs = 3.0


    def manage_request(self, request, socket):
        if request == "test":
            time.sleep(1)
            return b'Hi there', False
        elif request == "handshake":
            return (json.dumps(self.model_list), True)
        elif request == "test_tensor":
            if self.model is None:
                # For simplicity it just loads an arbitrary model.
                self.model = keras.models.load_model('saved_models/simple_mlp.keras')
                (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.mnist.load_data()
                #return b'No model set! Please load a model before calling this function.', False

            json_responses = []
            for layer_name in ['dense', 'dense_1']:
                print('Processed layer {}'.format(layer_name))
                layer = self.model.get_layer(layer_name)
                w = layer.get_weights()
                new_w = []
                for element in w:
                    new_w.append(element.tolist())
                json_str = json.dumps(new_w)
                json_responses.append(json_str)
            return json_responses, True
        elif request == "integrated_gradient":
            if self.model is None:
                return b'No model set yet', False
            if self.input is None:
                return b'No input set yet', False
            sample = np.expand_dims(self.input, 0)
            baseline = np.zeros_like(sample)
            ig = integrated_gradients(self.model, baseline, sample)
            ig = ig.numpy().reshape((28, 28))
            json_w = []
            for element in ig:
                json_w.append(element.tolist())
            json_str = json.dumps(json_w)
            return json_str, True
        elif request == "load_model":
            response = b'Received initial request. Now send the model name!'
            model_name = self.make_additional_requests(response, socket)
            print("Model Name received: {}".format(str(model_name)))
            if str(model_name) == "two_layer_mlp_relu_dead":
                self.model = keras.models.load_model('saved_models/' + model_name + '.keras', custom_objects={'custom_dead_relu_initializer': custom_objects.custom_dead_relu_initializer})
            else:
                self.model = keras.models.load_model('saved_models/' + model_name + '.keras')
            self.model_name = model_name
            (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.mnist.load_data()
            return b'Model loaded', False
        elif request == "load_simple_mlp":
            self.model = keras.models.load_model('saved_models/simple_mlp.keras')
            self.model_name = "simple_mlp"
            (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.mnist.load_data()
            return b'Simple MLP loaded', False
        elif request == "load_two_layer_mlp":
            self.model = keras.models.load_model('saved_models/two_layer_mlp.keras')
            self.model_name = "two_layer_mlp"
            (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.mnist.load_data()
            return b'Two Layer MLP loaded', False
        # TODO: This must depend on the model loaded previously - Edit: maybe not, if all have same train test split name convention
        # TODO: Add label capabilities. Both choosing by label and eventually storing label alongside the input
        elif request == "load_input":
            if self.model is None:
                return b'No model set yet', False
            response = b'Received initial request. Now send the set name!'
            input_set = self.make_additional_requests(response, socket)
            response = b'Received set name. Expecting the index next.'
            input_index = int(self.make_additional_requests(response, socket))

            if input_set == "train":
                self.input = self.x_train[input_index]
            elif input_set == "test":
                self.input = self.x_test[input_index]

            return b'Input Loaded successfully!', False
        elif request == "give_weights":
            if self.model is None:
                return b'No model set yet', False
            json_responses = []
            for layer in self.model.layers:
                w = layer.get_weights()
                if bool(w):
                    weights, biases = w
                    json_w = []
                    for element in weights:
                        json_w.append(element.tolist())
                    json_str = json.dumps(json_w)
                    json_responses.append(json_str)
                    json_b = []
                    for element in biases:
                        json_b.append(element.tolist())
                    json_str = json.dumps(json_b)
                    json_responses.append(json_str)
            return json_responses, True

        elif request == "send_activations":
            if self.model is None:
                return b'No model set yet', False
            if self.input is None:
                return b'No input set yet', False

            layer_names = list(map(lambda x: x.name, self.model.layers))

            output = self.model.get_layer(layer_names[0])(np.expand_dims(self.input, 0))
            json_responses = []

            i = 1
            while i < len(layer_names):
                output = self.model.get_layer(layer_names[i])(output)
                flattend_output = [item for sublist in output.numpy().tolist() for item in sublist]
                json_responses.append(flattend_output)
                i += 1

            return json_responses, True
        elif request == "send_class_average_activations":
            if self.model is None:
                return b'No model set yet', False

            with open('saved_precalculations/'+ self.model_name +'/class_average_activations.pickle', 'rb') as handle:
                saved_list = pickle.load(handle)

            return json.dumps(saved_list, default=convert_ndarray), True
        elif request == "send_class_average_signals":
            if self.model is None:
                return b'No model set yet', False

            response = "Message received. Now send the data_type used for the UMAP embedding"
            enum_value = int(self.make_additional_requests(response, socket))

            with open('saved_precalculations/'+ self.model_name +'/class_average_signals.pickle', 'rb') as handle:
                saved_list = pickle.load(handle)

            json_list = []
            json_list.append(json.dumps(saved_list, default=convert_ndarray))
            enum = UMAPData(enum_value)

            res = self.calculate_embeddings(enum, saved_list)
            print(res[0][0])
            json_list.append(json.dumps(res, default=convert_ndarray))

            return json_list, True
        elif request == "send_class_gradient_weighted_signals":
            if self.model is None:
                return b'No model set yet', False

            with open('saved_precalculations/'+ self.model_name +'/class_average_signals.pickle', 'rb') as handle:
                signals = pickle.load(handle)

            with open('saved_precalculations/'+ self.model_name +'/class_average_gradients.pickle', 'rb') as handle:
                gradients = pickle.load(handle)

            # Multiply each signal matrix of each class and layer with the corresponding gradient matrix#

            gradient_weighted_sigs = [[signals[i][j] * gradients[i][j] for j in range(len(signals[i]))] for i in
                                      range(len(signals))]

            return json.dumps(gradient_weighted_sigs, default=convert_ndarray), True
        elif request == "send_weighted_activations":
            if self.model is None:
                return b'No model set yet', False
            if self.input is None:
                return b'No input set yet', False


            layer_names = list(map(lambda x: x.name, self.model.layers))

            output = self.model.get_layer(layer_names[0])(np.expand_dims(self.input, 0))
            flattend_output = [item for sublist in output.numpy().tolist() for item in sublist]

            json_responses = []
            i = 1
            while i < len(layer_names):
                weights = self.model.get_layer(layer_names[i]).get_weights()[0]
                result = np.array([flattend_output] * weights.shape[1]).T * weights
                json_w = []
                for element in result:
                    json_w.append(element.tolist())
                output = self.model.get_layer(layer_names[i])(output)
                flattend_output = [item for sublist in output.numpy().tolist() for item in sublist]
                json_responses.append(json_w)
                i += 1

            return json_responses, True
        elif request == "send_average_activations":
            if self.model is None:
                return b'No model set yet', False
            if self.x_train is None:
                return b'Data not set yet', False

            with open('saved_precalculations/'+ self.model_name +'/average_activations.pickle', 'rb') as handle:
                layer_outputs = pickle.load(handle)

            json_w = []
            for element in layer_outputs:
                json_w.append(json.dumps(element.tolist()))

            return json_w, True
        elif request == "send_subset_activations":
            response = b'Received Request. Expecting the index next.'
            input_index = self.make_additional_requests(response, socket)
            if self.model is None:
                return b'No model set yet', False
            if self.x_train is None:
                return b'Data not set yet', False
            file_path = 'saved_precalculations/' + self.model_name + '/subset_activations_' + str(input_index) + '.pickle'
            with open(file_path, 'rb') as handle:
                layer_outputs = pickle.load(handle)
            return json.dumps(layer_outputs, default=convert_ndarray), True
        elif request == "send_average_signals":
            if self.model is None:
                return b'No model set yet', False
            if self.x_train is None:
                return b'Data not set yet', False

            file_path = 'saved_precalculations/' + self.model_name + '/average_signals.pickle'
            with open(file_path, 'rb') as handle:
                average_signal_per_layer = pickle.load(handle)

            json_w = []
            for element in average_signal_per_layer:
                python_list = []
                for i in range(0, element.shape[0], 100):
                    chunk = element[i:i + 100].tolist()
                    python_list.extend(chunk)
                json_w.append(json.dumps(python_list))

            return json_w, True
        elif request == "send_class_predictions_activations_and_sigs":
            if self.model is None:
                return b'No model set yet', False

            json_list = []

            file_path = 'saved_precalculations/' + self.model_name + '/class_correct_average_activations.pickle'
            with open(file_path, 'rb') as handle:
                saved_list = pickle.load(handle)
            json_list.append(json.dumps(saved_list, default=convert_ndarray))

            file_path = 'saved_precalculations/' + self.model_name + '/class_incorrect_average_activations.pickle'
            with open(file_path, 'rb') as handle:
                saved_list = pickle.load(handle)
            json_list.append(json.dumps(saved_list, default=convert_ndarray))

            file_path = 'saved_precalculations/' + self.model_name + '/class_correct_average_signals.pickle'
            with open(file_path, 'rb') as handle:
                saved_list = pickle.load(handle)
            json_list.append(json.dumps(saved_list, default=convert_ndarray))

            file_path = 'saved_precalculations/' + self.model_name + '/class_incorrect_average_signals.pickle'
            with open(file_path, 'rb') as handle:
                saved_list = pickle.load(handle)
            json_list.append(json.dumps(saved_list, default=convert_ndarray))

            return json_list, True
        elif request == "send_naps":
            #TODO: Make up concept of NWPs. Choose a different role / meaning than NAPs


            # Check if model was loaded before requesting naps
            if self.model is None:
                return b'No model set yet', False
            # Send confirmation response
            response = b'Received initial request. Ready to read the layers'
            decoded_message = self.make_additional_requests(response, socket)
            layers = json.loads(decoded_message)
            print("Received layers: %s" % layers)

            nwps = get_NWPs(self.model, 'MNIST', layers, 1000, None)

            json_responses = []
            for element in nwps['values']:
                tmp_nwps = nwps['values'][element].tolist()
                json_str = json.dumps(tmp_nwps)
                json_responses.append(json_str)

            return json_responses, True
        elif request == "send_input":
            if self.model is None:
                return b'No model set yet', False
            if self.input is None:
                return b'No input set yet', False

            json_response = json.dumps(self.input.tolist())
            return json_response, True

        else:
            print("Request {} not found. Make sure you added all enums to the python side!".format(request))
            return b'Request unknown', False

    # Function that asks the client for additional params, and returns said params decoded as a UTF-8 string
    def make_additional_requests(self, message, socket):
        socket.send_string(str(message))
        # Wait for next message to receive layers that are supposed to be processed
        message = None
        time_start = time.time()
        while message is None:
            message = socket.recv()
            current_time = time.time()
            if (current_time - time_start) > self.time_until_timeout_secs:
                print("Timeout occurred.")
                response = b'Timeout, please send another request'
                socket.send(response)
                return
        decoded_message = message.decode('utf-8')
        return decoded_message

    def calculate_embeddings(self, enum, class_average_signals):
        if enum == UMAPData.input:
            print("Using Input as Data")
            layerwise_neuron_data = []
            for index in range(0, len(class_average_signals[0]) - 1):
                layer_array = np.transpose(np.stack([class_average_signals[i][index] for i in range(10)]), [2, 1, 0])
                layer_array = layer_array.reshape([layer_array.shape[0], np.prod(layer_array.shape[1:])])
                layerwise_neuron_data.append(layer_array)
        elif enum == UMAPData.output:
            print("Using Output as Data")
            layerwise_neuron_data = []
            for index in range(1, len(class_average_signals[0])):
                layer_array = np.transpose(np.stack([class_average_signals[i][index] for i in range(10)]), [1, 2, 0])
                layer_array = layer_array.reshape([layer_array.shape[0], np.prod(layer_array.shape[1:])])
                layerwise_neuron_data.append(layer_array)
        elif enum == UMAPData.input_and_output:
            print("Using Input and Output as Data")
            layerwise_neuron_data = []
            transposed_array = [
                [np.transpose(class_average_signals[i][j]) for j in range(len(class_average_signals[0]))] for i in
                range(len(class_average_signals))]
            for index in range(0, len(class_average_signals[0]) - 1):
                layer_array = np.transpose(np.stack(
                    [np.concatenate((transposed_array[i][index], class_average_signals[i][index + 1]), axis=1) for i in
                     range(10)]), [1, 2, 0])
                layer_array = layer_array.reshape([layer_array.shape[0], np.prod(layer_array.shape[1:])])
                layerwise_neuron_data.append(layer_array)
        elif enum == UMAPData.activations:
            print("Using Activations as Data")
            with open('saved_precalculations/' + self.model_name + '/class_average_activations.pickle', 'rb') as handle:
                raw_data = pickle.load(handle)
            layerwise_neuron_data = [np.transpose(np.stack([raw_data[i][j] for i in range(len(raw_data))])) for j in range(len(raw_data[0]))][1:-1]
        elif enum == UMAPData.subset_activations:
            print("Using Subset Activations as Data")
            file_path = 'saved_precalculations/' + self.model_name + '/subset_activations_' + str(
                1) + '.pickle'
            with open(file_path, 'rb') as handle:
                layerwise_neuron_data = pickle.load(handle)
            # Cut off input and output layers
            layerwise_neuron_data = layerwise_neuron_data[1:-1]
        elif enum == UMAPData.subset_signals_output:
            print("Using Subset Output Signals as Data")
            with open('saved_precalculations/' + self.model_name + '/subset_signals.pickle', 'rb') as handle:
                raw_data = pickle.load(handle)
            reshaped_results = [raw_data[i].reshape(len(raw_data[i]),
                                                                      len(raw_data[i][0]) * len(
                                                                          raw_data[i][0][0])) for i in
                                     range(len(raw_data))]
            # Cut off input layer neurons
            layerwise_neuron_data = reshaped_results[1:]
        elif enum == UMAPData.subset_signals_input:
            print("Using Subset Input Signals as Data")
            with open('saved_precalculations/' + self.model_name + '/subset_signals.pickle', 'rb') as handle:
                raw_data = pickle.load(handle)

            reshaped_results = [raw_data[i].reshape(len(raw_data[i][0]),
                                                                      len(raw_data[i]) * len(
                                                                          raw_data[i][0][0])) for i in
                                     range(len(raw_data))]
            # Cut off output layer neurons
            layerwise_neuron_data = reshaped_results[:-1]
        elif enum == UMAPData.subset_signals_input_and_output:
            print("Using Subset Input and Output Signals as Data")
            with open('saved_precalculations/' + self.model_name + '/subset_signals.pickle', 'rb') as handle:
                raw_data = pickle.load(handle)
            reshaped_results = [raw_data[i].reshape(len(raw_data[i]),
                                                    len(raw_data[i][0]) * len(
                                                        raw_data[i][0][0])) for i in
                                range(len(raw_data))]
            # Cut off input layer neurons
            output_data = reshaped_results[1:]

            reshaped_results = [raw_data[i].reshape(len(raw_data[i][0]),
                                                    len(raw_data[i]) * len(
                                                        raw_data[i][0][0])) for i in
                                range(len(raw_data))]
            # Cut off output layer neurons
            input_data = reshaped_results[:-1]

            layerwise_neuron_data = [np.concatenate((output_data[i], input_data[i]), axis=1) for i in
                                     range(len(output_data))]
        else:
            print("Unknown enum '{}'!".format(enum))
            return

        embeddings_list = []
        for layer in layerwise_neuron_data:
            raw_embeddings = UMAP(n_components=2, metric="euclidean").fit_transform(layer)
            # Center embeddings at zero
            zero_centered_embeddings = raw_embeddings - np.mean(raw_embeddings, axis=0)
            print(np.mean(zero_centered_embeddings, axis=0))
            embeddings_list.append(zero_centered_embeddings)

        return embeddings_list



