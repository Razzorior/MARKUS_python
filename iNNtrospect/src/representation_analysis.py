import json
import os

import numpy as np
from umap import UMAP

from iNNtrospect.constants.check_constants import PIPELINE_STEPS
from iNNtrospect.constants.directory_constants import OUTPUT_DIRECTORY_NAMES
from iNNtrospect.src.checks import check_pipeline_dependencies, check_group_names_of_interest
from iNNtrospect.src.data_processing import get_n_layers
from iNNtrospect.src.utils import makedirs


def stack_and_flat_batch_activations(processed_corpus_path, output_dir, group_names_of_interest, layer, n_batches_per_group):
    activations = []
    instance_group_names = []

    with open(os.path.join(processed_corpus_path, "group_name_to_files.json"), "r") as f:
        group_name_to_files = json.load(f)

    for key in group_names_of_interest:
        if key in group_name_to_files.keys():
            batch_names = group_name_to_files[key][:n_batches_per_group]
            for batch_name in batch_names:
                batch = np.load(os.path.join(processed_corpus_path, OUTPUT_DIRECTORY_NAMES.ACTS, "layer" + str(layer).zfill(3), batch_name))
                activations.append(batch)
                instance_group_names.append(np.array(len(batch) * [key]))

    activations = np.concatenate(activations, 0)
    flat_activations = np.reshape(activations, [activations.shape[0],
                                                np.prod(activations.shape[1:])])
    instance_group_names = np.concatenate(instance_group_names,0)

    np.save(os.path.join(output_dir, "layer" + str(layer).zfill(3) + "_flat_acts.npy"), flat_activations)
    np.save(os.path.join(output_dir, "layer" + str(layer).zfill(3) + "_inst_group_names.npy"),
            instance_group_names)

def compute_projection_from_flat_activations(output_dir, layer):
    flat_activations = np.load(os.path.join(output_dir, "layer" + str(layer).zfill(3) + "_flat_acts.npy"))

    projection = UMAP(n_components=2).fit_transform(flat_activations)

    np.save(os.path.join(output_dir, "layer" + str(layer).zfill(3) + "_projection.npy"), projection)

def compute_instance_projections(processed_corpus_path, group_names_of_interest, n_batches_per_group):
    check_pipeline_dependencies(processed_corpus_path, PIPELINE_STEPS.INSTANCE_PROJECTION)

    group_names_of_interest = check_group_names_of_interest(processed_corpus_path, group_names_of_interest)

    projection_output_dir = os.path.join(processed_corpus_path, OUTPUT_DIRECTORY_NAMES.INSTANCE_PROJECTION_DATA)
    makedirs([projection_output_dir])

    n_layers = get_n_layers(processed_corpus_path)

    for layer in range(n_layers - 1):
        stack_and_flat_batch_activations(processed_corpus_path,
                                         projection_output_dir,
                                         group_names_of_interest,
                                         layer,
                                         n_batches_per_group)

        compute_projection_from_flat_activations(projection_output_dir,
                                                 layer)


