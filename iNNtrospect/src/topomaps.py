# functions to compute topographic layouts of values
import json
import os.path
import shutil

import numpy as np
from umap import UMAP
from sklearn.manifold import TSNE

from iNNtrospect.constants.check_constants import PIPELINE_STEPS
from iNNtrospect.constants.directory_constants import OUTPUT_DIRECTORY_NAMES
from iNNtrospect.src.checks import check_pipeline_dependencies
from iNNtrospect.src.data_processing import get_n_layers
from iNNtrospect.src.utils import makedirs, tanh, zero_one_feature_scaling


def compute_UMAP(values):
    return UMAP(n_components=2).fit_transform(values)

def compute_TSNE(values):
    return TSNE(init='pca', learning_rate='auto', n_components=2).fit_transform(values)

def distribute_to_circle(pos_init, n_steps=100):

    positions = np.copy(pos_init)

    f_glob = 0

    a_loc_att = 1.5
    b_loc_att = 15
    c_loc_att = 2

    i_cont = np.linspace(-3, 6, n_steps)

    for i in range(0, n_steps):
        # train step of the swarm
        f_g_coeff = tanh(i_cont[i], inv=True)
        f_l_coeff = tanh(i_cont[i])

        pairwise_differences = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]  # shape = (128,128,2)
        pairwise_distances = np.linalg.norm(pairwise_differences, axis=-1)  # (128,128)

        pairwise_distances[(pairwise_distances < 0.01)] = 0.01

        attract = a_loc_att * (1 / (pairwise_distances + 1) ** 3)
        repulse = b_loc_att * np.exp(-(pairwise_distances / c_loc_att))  # bounded repulsion; shape=(128,128)
        f = (attract - repulse)  # shape=(128,128)

        f = ((f * f_l_coeff) + (f_glob * f_g_coeff)) / 2

        x_ = pairwise_differences[:, :, 0] * f  # shape=(128,128)
        y_ = pairwise_differences[:, :, 1] * f
        x_ = x_.mean(axis=1)  # (128,)
        y_ = y_.mean(axis=1)
        x_ = x_[..., np.newaxis]  # (128,1)
        y_ = y_[..., np.newaxis]
        update = np.concatenate((x_, y_), axis=1)  # shape=(128, 2)
        # update = np.nan_to_num(update)
        positions += update

        # add little noise to avoid points falling onto the same location
        positions += np.random.normal(0,0.000001, size=positions.shape)

    return positions


def get_nap_dir(processed_corpus_path, from_contrastive_naps):
    if from_contrastive_naps is None:
        if os.path.isdir(os.path.join(processed_corpus_path, OUTPUT_DIRECTORY_NAMES.CONTRASTIVE_NAPS)):
            from_contrastive_naps = True
        else:
            from_contrastive_naps = False

    if from_contrastive_naps:
        nap_dir = os.path.join(processed_corpus_path, OUTPUT_DIRECTORY_NAMES.CONTRASTIVE_NAPS)
    else:
        nap_dir = os.path.join(processed_corpus_path, OUTPUT_DIRECTORY_NAMES.NAPS)

    return nap_dir

def compute_and_save_layer_topomap_layouts(values_dir, output_dir, layer, layouting_function, distribute_in_circle):

    layer_id = "layer" + str(layer).zfill(3)

    nap = np.load(os.path.join(values_dir, layer_id + ".npy"))
    nap_shape = nap.shape

    if nap_shape[-1] > 15:
        # flattened profile per channel - each column contains nap values of one output channel, all groups stacked
        flat_channel_profile = np.transpose(np.reshape(nap,
                                                       [np.prod(nap_shape[:-1]), nap_shape[-1]]
                                                       ))
        # remove channel positions which are (almost) zero everywhere
        flat_channel_profile = flat_channel_profile[:, np.max(np.abs(flat_channel_profile), 0) > 0.01]

        coordinates = layouting_function(flat_channel_profile)
        coordinates = zero_one_feature_scaling(coordinates)

        if distribute_in_circle:
            coordinates = distribute_to_circle(coordinates)
            coordinates = zero_one_feature_scaling(coordinates)

        output_path = os.path.join(output_dir, layer_id + "_coordinates.npy")

        np.save(output_path, coordinates)
    else:
        print("skipping", layer_id, "- too few channels to compute layout for")


def compute_topomap_layout(processed_corpus_path,
                           layouting_method="UMAP",
                           distribute_in_circle=True,
                           from_contrastive_naps=None):

    if from_contrastive_naps:
        check_pipeline_dependencies(processed_corpus_path, PIPELINE_STEPS.TOPOMAP_LAYOUT_CONTRASTIVE)
    else:
        check_pipeline_dependencies(processed_corpus_path, PIPELINE_STEPS.TOPOMAP_LAYOUT)

    n_layers = get_n_layers(processed_corpus_path)
    nap_dir = get_nap_dir(processed_corpus_path, from_contrastive_naps)

    topomap_output_dir = os.path.join(processed_corpus_path, OUTPUT_DIRECTORY_NAMES.TOPOMAP_DATA)
    makedirs([topomap_output_dir])

    layouting_method_to_function = {"UMAP": compute_UMAP,
                                    "tSNE": compute_TSNE}

    layouting_function = layouting_method_to_function[layouting_method]

    for layer in range(n_layers - 1):
        compute_and_save_layer_topomap_layouts(nap_dir, topomap_output_dir, layer, layouting_function, distribute_in_circle)


def compute_and_save_layer_topomap_activations(values_dir, output_dir, layer):

    layer_id = "layer" + str(layer).zfill(3)

    nap = np.load(os.path.join(values_dir, layer_id + ".npy"))
    nap_shape = nap.shape

    if nap_shape[-1] > 1:
        while len(nap.shape) > 2:
            nap = np.mean(nap, 1)

        output_path = os.path.join(output_dir, layer_id + "_activations.npy")

        np.save(output_path, nap)
    else:
        print("skipping", layer_id, "- no channels to compute representative activation values for")


def compute_topomap_activations(processed_corpus_path, from_contrastive_naps=None):
    topomap_output_dir = os.path.join(processed_corpus_path, OUTPUT_DIRECTORY_NAMES.TOPOMAP_DATA)
    makedirs([topomap_output_dir])

    if from_contrastive_naps:
        check_pipeline_dependencies(processed_corpus_path, PIPELINE_STEPS.TOPOMAP_ACTIVATIONS_CONTRASTIVE)
        shutil.copyfile(os.path.join(processed_corpus_path, OUTPUT_DIRECTORY_NAMES.CONTRASTIVE_NAPS, "group_names.npy"),
                        os.path.join(topomap_output_dir, "group_names.npy"))
    else:
        check_pipeline_dependencies(processed_corpus_path, PIPELINE_STEPS.TOPOMAP_ACTIVATIONS)
        with open(os.path.join(processed_corpus_path, "group_name_to_index.json"), "r") as f:
            group_name_to_index = json.load(f)
        group_names = [*group_name_to_index.keys()]
        np.save(os.path.join(topomap_output_dir, "group_names.npy"),
                group_names)

    n_layers = get_n_layers(processed_corpus_path)
    nap_dir = get_nap_dir(processed_corpus_path, from_contrastive_naps)

    for layer in range(n_layers - 1):
        compute_and_save_layer_topomap_activations(nap_dir, topomap_output_dir, layer)