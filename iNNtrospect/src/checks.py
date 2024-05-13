import json
import numpy as np
import os.path
import shutil

from iNNtrospect.constants.directory_constants import OUTPUT_DIRECTORY_NAMES

pipeline_dir_input_dependencies = {
    "acts_grads": [],
    "align": [OUTPUT_DIRECTORY_NAMES.ACTS,
              OUTPUT_DIRECTORY_NAMES.GRADS],
    "naps": [OUTPUT_DIRECTORY_NAMES.ACTS],
    "naps_aligned": [OUTPUT_DIRECTORY_NAMES.ALIGNED],
    "instance_projection": [OUTPUT_DIRECTORY_NAMES.ACTS],
    "instance_projection_plots": [OUTPUT_DIRECTORY_NAMES.INSTANCE_PROJECTION_DATA],
    "contrastive_naps": [OUTPUT_DIRECTORY_NAMES.NAPS],
    "topomap_layout": [OUTPUT_DIRECTORY_NAMES.NAPS],
    "topomap_layout_contrastive": [OUTPUT_DIRECTORY_NAMES.CONTRASTIVE_NAPS],
    "topomap_activations": [OUTPUT_DIRECTORY_NAMES.NAPS],
    "topomap_activations_contrastive": [OUTPUT_DIRECTORY_NAMES.CONTRASTIVE_NAPS],
    "topomap_plots": [OUTPUT_DIRECTORY_NAMES.TOPOMAP_DATA]
}

assure_sync = True

pipeline_dir_async_effects = {
    "acts_grads": [OUTPUT_DIRECTORY_NAMES.ACTS,
                   OUTPUT_DIRECTORY_NAMES.GRADS,
                   OUTPUT_DIRECTORY_NAMES.ALIGNED,
                   OUTPUT_DIRECTORY_NAMES.NAPS,
                   OUTPUT_DIRECTORY_NAMES.CONTRASTIVE_NAPS,
                   OUTPUT_DIRECTORY_NAMES.INSTANCE_PROJECTION_DATA,
                   OUTPUT_DIRECTORY_NAMES.INSTANCE_PROJECTION_PLOTS,
                   OUTPUT_DIRECTORY_NAMES.TOPOMAP_DATA,
                   OUTPUT_DIRECTORY_NAMES.TOPOMAP_PLOTS],
    "align": [OUTPUT_DIRECTORY_NAMES.ALIGNED,
              OUTPUT_DIRECTORY_NAMES.NAPS,
              OUTPUT_DIRECTORY_NAMES.CONTRASTIVE_NAPS,
              OUTPUT_DIRECTORY_NAMES.INSTANCE_PROJECTION_DATA,
              OUTPUT_DIRECTORY_NAMES.INSTANCE_PROJECTION_PLOTS,
              OUTPUT_DIRECTORY_NAMES.TOPOMAP_DATA,
              OUTPUT_DIRECTORY_NAMES.TOPOMAP_PLOTS],
    "naps": [OUTPUT_DIRECTORY_NAMES.NAPS,
             OUTPUT_DIRECTORY_NAMES.CONTRASTIVE_NAPS,
             OUTPUT_DIRECTORY_NAMES.TOPOMAP_DATA,
             OUTPUT_DIRECTORY_NAMES.TOPOMAP_PLOTS],
    "naps_aligned": [OUTPUT_DIRECTORY_NAMES.NAPS,
                     OUTPUT_DIRECTORY_NAMES.CONTRASTIVE_NAPS,
                     OUTPUT_DIRECTORY_NAMES.TOPOMAP_DATA,
                     OUTPUT_DIRECTORY_NAMES.TOPOMAP_PLOTS],
    "instance_projection": [OUTPUT_DIRECTORY_NAMES.INSTANCE_PROJECTION_DATA],
    "instance_projection_plots": [OUTPUT_DIRECTORY_NAMES.INSTANCE_PROJECTION_PLOTS],
    "contrastive_naps": [OUTPUT_DIRECTORY_NAMES.TOPOMAP_DATA,
                         OUTPUT_DIRECTORY_NAMES.TOPOMAP_PLOTS],
    "topomap_layout": [OUTPUT_DIRECTORY_NAMES.TOPOMAP_PLOTS],
    "topomap_layout_contrastive": [OUTPUT_DIRECTORY_NAMES.TOPOMAP_PLOTS],
    "topomap_activations": [OUTPUT_DIRECTORY_NAMES.TOPOMAP_PLOTS],
    "topomap_activations_contrastive": [OUTPUT_DIRECTORY_NAMES.TOPOMAP_PLOTS],
    "topomap_plots": []
}

def get_user_delete_agreement():
    print('Do you agree with deleting potentially dependent directories? (yes/no):')
    x = input()
    if x != "yes":
        stop_msg = "Stopping: either agree to deleting or set 'assure_sync=False' in src/checks.py ."
        raise ValueError(stop_msg)
    else:
        return True

def check_pipeline_dependencies(processed_corpus_path, step):

    for dir_name in pipeline_dir_input_dependencies[step]:
        check_path = os.path.join(processed_corpus_path, dir_name)
        if not os.path.isdir(check_path):
            err_msg = "step '" + step + "' depends on '" + dir_name + "' which does not exist. "
            raise ValueError(err_msg)

    if assure_sync:
        agreed_deleting = False
        for dir_name in pipeline_dir_async_effects[step]:
            check_path = os.path.join(processed_corpus_path, dir_name)
            if os.path.isdir(check_path):
                info_msg = ("running step '" + step + "' overwrites data that was used for creating '" + dir_name + "'.\n" +
                            "Needs to be deleted according to 'assure_sync=True' flag in src/checks.py .")
                print(info_msg)
                if not agreed_deleting:
                    agreed_deleting = get_user_delete_agreement()
                shutil.rmtree(check_path)

def check_group_names_of_interest(processed_corpus_path, group_names_of_interest):
    with open(os.path.join(processed_corpus_path, "group_name_to_index.json"), "r") as f:
        group_name_to_index = json.load(f)

    if group_names_of_interest is None:
        group_names_of_interest = [*group_name_to_index.keys()]
    else:
        group_names_of_interest = np.unique(group_names_of_interest)

    return group_names_of_interest