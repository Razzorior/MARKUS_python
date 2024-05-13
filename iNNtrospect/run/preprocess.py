# script for running a preprocessing pipeline to generate data for iNNtrospection
# fixed instances of this script go to examples

# from env_setup import local_env_settings
# local_env_settings()

import json
from iNNtrospect.src.data_processing import process_corpus_file, align_data
from iNNtrospect.src.model import load_model, create_model_with_layer_outputs
from iNNtrospect.src.naps import compute_naps

config_file = "examples/configs/MNIST_MLP.json"
# config_file = "examples/configs/MNIST_CNN_SHALLOW.json"
with open(config_file, "r") as f:
    config = json.load(f)

# load model
model = load_model(config["model_path"])

# make model with output layers of interest
model_with_layer_outputs = create_model_with_layer_outputs(model,
                                                           config["layer_names_of_interest"])

# provide (batches) of examples with group annotation
# format:
# filename, groupname, (expected output)
# batch1.npy, shirt, (0)
# batch2.npy, shirt, (0)
# batch3.npy, trouser, (1)

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




