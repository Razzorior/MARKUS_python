# from env_setup import local_env_settings
# local_env_settings()

from constants.topomap_constants import TOPOMAP_METHODS, TOPOMAP_PLOTTING_MODES
from src.data_processing import process_corpus_file, align_data
from src.model import load_model, create_model_with_layer_outputs
from src.naps import compute_naps, compute_contrastive_naps
from src.representation_analysis import compute_instance_projections
from src.topomaps import compute_topomap_layout, compute_topomap_activations
from src.visualization import plot_topomaps, plot_instance_projections

import os
import json
import numpy as np
import tensorflow as tf

config_file = "examples/MNIST/configs/MNIST_MLP.json"
# config_file = "examples/MNIST/configs/MNIST_CNN_SHALLOW.json"
with open(config_file, "r") as f:
    config = json.load(f)

'''
Preprocessing Evaluation Data

Create batches of data examples from the same group of interest.
Create a corpus.csv that maps batch file name to the group name and (optionally) the output id 
# format:
# batch1.npy, shirt, (0)
# batch2.npy, shirt, (0)
# batch3.npy, trouser, (1)
'''

output_path = config["data_path"]
batch_size = 128
n_random_per_class = 200

if not os.path.isdir(output_path):
    os.makedirs(output_path)

(_, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

test_images_rescaled = (test_images.astype('float32') / 255.)[..., np.newaxis]

corpus_file_content = []
batch_id = 0
for cid in range(10):
    class_example_ids = np.argwhere(test_labels==cid)[:,0]

    random_class_ids = class_example_ids[np.random.choice(len(class_example_ids),
                                                          n_random_per_class)]

    batch_start_idx = 0

    while batch_start_idx < n_random_per_class:
        image_batch = test_images_rescaled[random_class_ids[batch_start_idx:batch_start_idx+batch_size]]

        batch_name = 'batch' + str(batch_id).zfill(4)
        np.save(os.path.join(output_path, batch_name), image_batch)
        corpus_file_content.append([batch_name+'.npy', str(cid), cid])

        batch_start_idx = batch_start_idx + batch_size
        batch_id = batch_id + 1

corpus_file_content = np.array(corpus_file_content)
np.savetxt(os.path.join(output_path, 'corpus.csv'), corpus_file_content, delimiter=",", fmt='%s')


'''
Main Processing Steps

For a given model and the preprocessed batches, compute 
- activations in layers of interest
- gradients in layers of interest
- alignment of activations
- basic NAP values (group averages of activations)
'''

model = load_model(config["model_path"])

# make model with output layers of interest
model_with_layer_outputs = create_model_with_layer_outputs(model,
                                                           config["layer_names_of_interest"])


# computing activations and gradients
# saved to <processed_corpus_path>/acts/ and <processed_corpus_path>/grads/
process_corpus_file(config["data_path"],
                    model_with_layer_outputs,
                    config["processed_corpus_path"])

# creating aligned activations in every layer where possible
# saved to <processed_corpus_path>/aligned/
if config["use_aligned_naps"]:
    align_data(config["processed_corpus_path"])

# compute group averages in each layer
# saved to <processed_corpus_path>/naps/
compute_naps(config["processed_corpus_path"], use_aligned_acts=config["use_aligned_naps"])


'''
Introspection based on NAPs

here including
- 2D projection of instances based on layer activations
- computing contrastive NAPs
- creating topographic map layouts
- plotting topographic maps
'''

group_names_of_interest_projection = None #["0","2","4","9"]

compute_instance_projections(config["processed_corpus_path"],
                             group_names_of_interest = group_names_of_interest_projection,
                             n_batches_per_group = 1)

plot_instance_projections(config["processed_corpus_path"])


group_names_of_interest_naps = None #["0","2","4","9"]
weight_by_group_size = False
compute_contrastive_naps(config["processed_corpus_path"],
                         group_names_of_interest = group_names_of_interest_naps,
                         weight_by_group_size = weight_by_group_size)

compute_topomap_layout(config["processed_corpus_path"],
                       layouting_method = TOPOMAP_METHODS.UMAP,
                       distribute_in_circle = True,
                       from_contrastive_naps = True)

compute_topomap_activations(config["processed_corpus_path"],
                            from_contrastive_naps = True)

plot_topomaps(config["processed_corpus_path"],
              mode=TOPOMAP_PLOTTING_MODES.SINGLE)
plot_topomaps(config["processed_corpus_path"],
              mode=TOPOMAP_PLOTTING_MODES.ROW)



