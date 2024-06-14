def calc_class_average_activation_and_signals(model, file_path_activation=None, file_path_signals=None):
    (x_train_unnormalized, y_train), (x_test_unnormalized, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train_unnormalized / 255.0
    x_test = x_test_unnormalized / 255.0
    batch_size = 32

    functions = []
    transposed_weights = []

    class_average_signals_per_layer = []
    class_average_activations_per_layer = []

    for i in range(len(model.layers)):

        functions.append(K.function([model.layers[i].input],model.layers[i].output))
        if len(model.layers[i].get_weights()) == 0:
            transposed_weights.append([])
        else:
            transposed_weights.append(tf.transpose(model.layers[i].get_weights()[0]))

    # 10 is hardcoded for the amount of MNIST classes. In case there is a different model we need to change this part (and the dataloading of course)
    for class_index in range (10):
        class_images = x_train[y_train==class_index]
        dataset = tf.data.Dataset.from_tensor_slices(class_images).batch(batch_size)
        class_activations = []
        class_signals = []
        for i, batch in enumerate(dataset):
            output = functions[0](batch)
            if i == 0:
                class_activations.append(tf.expand_dims(tf.reduce_sum(output, axis=0), axis=0))
            else:
                class_activations[0] = tf.add(class_activations[0], tf.expand_dims(tf.reduce_sum(output, axis=0), axis=0))
                #class_activations[0] = tf.add([class_activations[0], tf.expand_dims(tf.reduce_sum(output, axis=0), axis=0)])

            for j in range(1, len(functions)):
                output = functions[j](output)
                if i == 0:
                    class_activations.append(tf.expand_dims(tf.reduce_sum(output, axis=0), axis=0))
                    class_signals.append(tf.expand_dims(tf.reduce_sum(tf.einsum('bi,ij->bji', output, transposed_weights[j]),axis=0), axis=0))
                else:
                    class_activations[j] = tf.add(class_activations[j], tf.expand_dims(tf.reduce_sum(output, axis=0), axis=0))
                    class_signals[j-1] = tf.add(class_signals[j-1], tf.expand_dims(tf.reduce_sum(tf.einsum('bi,ij->bji', output, transposed_weights[j]),axis=0), axis=0))

        print("Done with class {}..".format(str(class_index)))

        elements_in_class = class_images.shape[0]

        #for index in range(len(class_activations)):
        #    class_activations[index] = tf.reduce_mean(class_activations[index], axis=0).numpy()

        #for index in range(len(class_signals)):
        #    class_signals[index] = tf.reduce_mean(class_signals[index], axis=0).numpy()


        class_average_activations_per_layer.append(class_activations)
        class_average_signals_per_layer.append(class_signals)

    print("Computing overall activation averages..")
    average_activations_per_layer = []
    num_layers = len(class_average_activations_per_layer[0])
    for layer_index in range(num_layers):
        # Initialisieren Sie einen Tensor, um die summierten Aktivierungen für die aktuelle Schicht zu speichern
        summed_activations = tf.zeros_like(class_average_activations_per_layer[0][layer_index])

        # Addieren Sie die durchschnittlichen Aktivierungen der aktuellen Schicht über alle Klassen hinweg
        for class_activations in class_average_activations_per_layer:
            summed_activations = tf.add(summed_activations, class_activations[layer_index])

        # Fügen Sie den Durchschnitt der aktuellen Schicht zur Ergebnisliste hinzu
        average_activations_per_layer.append(summed_activations)

    for layer_index in range(num_layers):
        average_activations_per_layer[layer_index] = np.squeeze(tf.divide(average_activations_per_layer[layer_index], x_train.shape[0]).numpy())

    print("Averaging activation classes..")
    for class_index, class_activations in enumerate(class_average_activations_per_layer):
        for layer_index, layer in enumerate(class_activations):
            class_average_activations_per_layer[class_index][layer_index] = np.squeeze((tf.divide(layer[0], np.sum(y_train == class_index))).numpy())

    print("Computing overall signal averages..")
    average_signals_per_layer = []
    num_layers = len(class_average_signals_per_layer[0])
    for layer_index in range(num_layers):
        summed_activations = tf.zeros_like(class_average_signals_per_layer[0][layer_index])

        for class_activations in class_average_signals_per_layer:
            summed_activations = tf.add(summed_activations, class_activations[layer_index])

        average_signals_per_layer.append(summed_activations)

    for layer_index in range(num_layers):
        average_signals_per_layer[layer_index] = np.squeeze(tf.divide(average_signals_per_layer[layer_index], x_train.shape[0]).numpy())

    print("Averaging signal classes..")
    for class_index, class_signals in enumerate(class_average_signals_per_layer):
        for layer_index, layer in enumerate(class_signals):
            class_average_signals_per_layer[class_index][layer_index] = np.squeeze(tf.divide(layer[0], np.sum(y_train == class_index)).numpy())

    total_average_activations = []
    total_average_signals = []

    print("Saving results..")
    return class_average_activations_per_layer, class_average_signals_per_layer, average_activations_per_layer, average_signals_per_layer