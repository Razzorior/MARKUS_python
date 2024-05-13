import json
import os
import shutil

import tensorflow as tf
import numpy as np

from iNNtrospect.constants.check_constants import PIPELINE_STEPS
from iNNtrospect.constants.directory_constants import OUTPUT_DIRECTORY_NAMES
from iNNtrospect.src.checks import check_pipeline_dependencies
from iNNtrospect.src.utils import makedirs


def create_acts_grads_output_dirs(output_path, n_layers, with_aligned=False):
    output_acts_path = os.path.join(output_path, OUTPUT_DIRECTORY_NAMES.ACTS)
    output_grads_path = os.path.join(output_path, OUTPUT_DIRECTORY_NAMES.GRADS)
    output_aln_path = None
    if with_aligned:
        output_aln_path = os.path.join(output_path, OUTPUT_DIRECTORY_NAMES.ALIGNED)
    for i in range(n_layers):
        layer_running_id = 'layer' + str(i).zfill(3)
        layer_output_acts_path = os.path.join(output_acts_path, layer_running_id)
        layer_output_grads_path = os.path.join(output_grads_path, layer_running_id)
        if i == n_layers-1:
            paths_to_make = [layer_output_acts_path]
        else:
            paths_to_make = [layer_output_acts_path, layer_output_grads_path]
        if with_aligned and i < n_layers-1:
            layer_output_aln_path = os.path.join(output_aln_path, layer_running_id)
            paths_to_make.append(layer_output_aln_path)
        makedirs(paths_to_make)

def get_acts_and_grads_of_batch(model, batch):
    batch = tf.convert_to_tensor(batch)

    with tf.GradientTape() as tape:
        tape.watch(batch)
        batch_activations = model(batch)

        outputs = batch_activations[-1]
        preds = np.argmax(outputs, 1)
        pred_logits = [o[p].numpy() for o, p in zip(outputs, preds)]

        batch_grads = tape.gradient(batch_activations[-1],
                                    batch_activations[:-1],
                                    output_gradients=tf.one_hot(preds, outputs.shape[-1]))
    return batch_activations, batch_grads

def save_batch_acts_and_grads_per_layer(acts, grads, batch_file, output_path):
    output_acts_path = os.path.join(output_path, OUTPUT_DIRECTORY_NAMES.ACTS)
    output_grads_path = os.path.join(output_path, OUTPUT_DIRECTORY_NAMES.GRADS)
    layer_of_interest_id = 0
    for layer_acts, layer_grads in zip(acts, grads):
        np.save(os.path.join(output_acts_path, 'layer' + str(layer_of_interest_id).zfill(3), batch_file),
                layer_acts)
        np.save(os.path.join(output_grads_path, 'layer' + str(layer_of_interest_id).zfill(3), batch_file),
                layer_grads)
        layer_of_interest_id = layer_of_interest_id + 1
    np.save(os.path.join(output_acts_path, 'layer' + str(layer_of_interest_id).zfill(3), batch_file),
            acts[-1])

def process_corpus_file(data_path, model, output_path):
    check_pipeline_dependencies(output_path, PIPELINE_STEPS.ACTS_GRADS)

    create_acts_grads_output_dirs(output_path, len(model.outputs))

    corpus_path = os.path.join(data_path, "corpus.csv")
    corpus = np.loadtxt(corpus_path, dtype=str, delimiter=',')

    group_names = np.unique(corpus[:, 1])

    group_to_file_dict = dict()
    for group_name in group_names:
        ids_of_group_files = np.argwhere(corpus[:,1] == group_name)[:,0]
        group_to_file_dict[group_name] = list(corpus[ids_of_group_files,0])

    with open(os.path.join(output_path,
                           'group_name_to_files.json'), 'w') as f:
        json.dump(group_to_file_dict, f)

    group_ids = np.arange(len(group_names)).astype('str')
    index_to_group_name = dict(zip(group_ids, group_names))
    with open(os.path.join(output_path, "index_to_group_name.json"), "w") as f:
        json.dump(index_to_group_name, f)
    group_name_to_index = dict(zip(group_names, group_ids))
    with open(os.path.join(output_path, "group_name_to_index.json"), "w") as f:
        json.dump(group_name_to_index, f)

    for batch_info in corpus:
        batch_file = batch_info[0]
        group_name = batch_info[1]
        label = None
        if len(batch_info) > 2:
            label = int(batch_info[2])

        batch = np.load(os.path.join(data_path, batch_file))
        batch_activations, batch_grads = get_acts_and_grads_of_batch(model,batch)
        save_batch_acts_and_grads_per_layer(batch_activations, batch_grads, batch_file, output_path)

def load_batch_acts_and_grads(processed_corpus_path, layer_id, batch_id):
    layer_batch_acts = np.load(os.path.join(processed_corpus_path,
                                            OUTPUT_DIRECTORY_NAMES.ACTS,
                                            'layer' + str(layer_id).zfill(3),
                                            'batch' + str(batch_id).zfill(4) + '.npy'))
    layer_batch_grads = np.load(os.path.join(processed_corpus_path,
                                             OUTPUT_DIRECTORY_NAMES.GRADS,
                                             'layer' + str(layer_id).zfill(3),
                                             'batch' + str(batch_id).zfill(4) + '.npy'))
    return layer_batch_acts, layer_batch_grads

def make_alignment_maps_from_gradients(grads):
    # if absolute: # sign of grad does not matter for importance
    alignment_map = np.abs(grads)
    # if average_channels: # consider importance across channels, not just one
    alignment_map = np.expand_dims(np.mean(alignment_map, -1), -1)
    return alignment_map

def align_by_pad_and_crop(activations, index_per_dim):
    # compute padding size necessary to shift the point of
    # maximum saliency to the center in each dimension
    pad_size = index_per_dim - (np.array(np.shape(activations)) // 2)

    # correspondingly pad with zeros on the edges closest to idx_to_align
    pad_slices = ()
    for i, p in enumerate(pad_size):
        #                     if i in dimensions_to_align:
        pad_slices += ((np.max((-p, 0)), np.max((p, 0))),)
    #                     else:
    #                         pad_slices += ((0, 0),)

    # apply padding
    padded_example_acts = np.pad(activations, pad_slices)

    # crop the same amount from the matrices at the opposite edges
    cropping_slices = ()
    for i, (p, d) in enumerate(zip(pad_size, padded_example_acts.shape)):
        #                     if i in dimensions_to_align:
        cropping_slices += (slice(np.max((p, 0)), d - np.max((-p, 0))),)
    #                     else:
    #                         cropping_slices += (slice(None),)

    # apply cropping
    aligned_example_acts = padded_example_acts[cropping_slices]
    return aligned_example_acts

def get_n_layers(processed_corpus_path):
    output_acts_path = os.path.join(processed_corpus_path, OUTPUT_DIRECTORY_NAMES.ACTS)
    layer_dirs = os.listdir(output_acts_path)
    layer_dirs = [l for l in layer_dirs if "layer" in l]
    n_layers = len(layer_dirs)

    return n_layers

def get_n_batches(processed_corpus_path):
    batch_filenames = os.listdir(os.path.join(processed_corpus_path, OUTPUT_DIRECTORY_NAMES.ACTS, "layer000"))
    batch_filenames = [bfn for bfn in batch_filenames if ".npy" in bfn]
    n_batches = len(batch_filenames)

    return n_batches

def align_data(processed_corpus_path):
    check_pipeline_dependencies(processed_corpus_path, PIPELINE_STEPS.ALIGN)
    n_layers = get_n_layers(processed_corpus_path)
    n_batches = get_n_batches(processed_corpus_path)

    create_acts_grads_output_dirs(processed_corpus_path, n_layers, with_aligned=True)

    for layer_id in range(n_layers-1):
        output_aligned_layer_path = os.path.join(processed_corpus_path, OUTPUT_DIRECTORY_NAMES.ALIGNED, 'layer' + str(layer_id).zfill(3))
        for batch_id in range(n_batches):
            layer_batch_acts, layer_batch_grads = load_batch_acts_and_grads(processed_corpus_path, layer_id, batch_id)

            # assumption: dim 0 is batch, dim -1 is channel
            # align all dims except batch and channel
            n_dims = len(layer_batch_acts.shape)
            if n_dims <= 2:
                print("layer" + str(layer_id).zfill(3) + " has no dimensions to align.")
                shutil.rmtree(output_aligned_layer_path)
                break
            else:
                aligned_layer_batch_acts = np.zeros_like(layer_batch_acts)
                alignment_map = make_alignment_maps_from_gradients(layer_batch_grads)

                # align every example in the batch
                for example_id in range(len(layer_batch_acts)):
                    example_acts = layer_batch_acts[example_id]
                    example_grads = alignment_map[example_id]

                    if np.min(example_grads) == np.max(example_grads):
                        aligned_layer_batch_acts[example_id] = example_acts
                    else:
                        index_to_align = np.array(np.unravel_index(np.argmax(example_grads), example_grads.shape))
                        aligned_example_acts = align_by_pad_and_crop(example_acts, index_to_align)
                        aligned_layer_batch_acts[example_id] = aligned_example_acts

                np.save(os.path.join(output_aligned_layer_path,
                                     'batch'+str(batch_id).zfill(4)+'.npy'),
                        aligned_layer_batch_acts)

