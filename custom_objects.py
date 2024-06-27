import tensorflow as tf
def custom_dead_relu_initializer(shape, dtype=None, partition_info=None):
    # Erzeugen von Gewichten, die zur Hälfte "tote" Neuronen verursachen
    alive_neurons = shape[1] // 2
    dead_neurons = shape[1] - alive_neurons

    # Positive Initialisierung für die "lebenden" Neuronen
    alive_weights = tf.random.normal((shape[0], alive_neurons), mean=0.1, stddev=0.05, dtype=dtype)

    # Negative Initialisierung für die "toten" Neuronen
    dead_weights = tf.random.normal((shape[0], dead_neurons), mean=-1.0, stddev=0.05, dtype=dtype)

    # Zusammenführen der Gewichte
    weights = tf.concat([alive_weights, dead_weights], axis=1)

    # Mischen der Gewichte, um eine zufällige Verteilung der "toten" Neuronen zu gewährleisten
    weights = tf.random.shuffle(weights, seed=42)

    return weights


class GradientMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Get the gradients of all the trainable weights
        for layer in self.model.layers:
            if hasattr(layer, 'kernel'):
                grads = self.model.optimizer.get_gradients(logs['loss'], layer.kernel)
                print(f'Gradients for layer {layer.name}: {grads}')