from constants.topomap_constants import TOPOMAP_METHODS, TOPOMAP_PLOTTING_MODES
from env_setup import local_env_settings
local_env_settings()

import json

from src.naps import compute_contrastive_naps
from src.representation_analysis import compute_instance_projections
from src.topomaps import compute_topomap_layout, compute_topomap_activations
from src.visualization import plot_instance_projections, plot_topomaps


config_file = "examples/FairFace/configs/fairface_VGG16.json"

with open(config_file, "r") as f:
    config = json.load(f)

group_names_of_interest_projection = None

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

# plot_topomaps(config["processed_corpus_path"],
#               mode=TOPOMAP_PLOTTING_MODES.SINGLE)
plot_topomaps(config["processed_corpus_path"],
              mode=TOPOMAP_PLOTTING_MODES.ROW)