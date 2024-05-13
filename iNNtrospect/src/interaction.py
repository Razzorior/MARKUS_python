# functions for I/O used in interaction via script or notebook
import json
import os

from iNNtrospect.src.data_processing import get_n_layers


def get_options_in_preprocessing_path(processed_corpus_path):

    dir_name = processed_corpus_path.split("/")[-1]
    n_layers = get_n_layers(processed_corpus_path)
    with open(os.path.join(processed_corpus_path, "group_name_to_index.json"), "r") as f:
        group_to_index_dict = json.load(f)
    group_names = [*group_to_index_dict.keys()]

    options_dict = {"name": dir_name,
                    "layers":["layer"+str(lid).zfill(3) for lid in range(n_layers-1)],
                    "group_names":group_names
                    }

    return options_dict