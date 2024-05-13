
from env_setup import local_env_settings
local_env_settings()

import json

from src.data_processing import process_corpus_file, align_data
from src.model import create_model_with_layer_outputs, load_imagenet_pretrained_model
from src.naps import compute_naps


config_file = "examples/FairFace/configs/fairface_VGG16.json"

with open(config_file, "r") as f:
    config = json.load(f)


model = load_imagenet_pretrained_model(config["model_name"])

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