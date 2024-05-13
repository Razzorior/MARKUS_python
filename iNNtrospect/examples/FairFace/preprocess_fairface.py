# from env_setup import local_env_settings
# local_env_settings()

from src.utils import makedirs
import os
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json

def image_path_to_numpy(file_name):
    img = Image.open(file_name)
    img.load()
    img_array = np.asarray(img, dtype="float32")
    return img_array

config_file = "examples/FairFace/configs/preprocess_fairface_race.json"
with open(config_file, "r") as f:
    config = json.load(f)

data_base_dir = config["raw_data_path"]
data_out_dir = config["processed_data_path"]

makedirs([data_out_dir])

protected_variable = config["variable"]
batch_size = 128
n_random_per_class = 10*batch_size

for data_split_subset in ["train"]:#, "val"]:
    raw_data_path = os.path.join(data_base_dir, data_split_subset)
    onlyfiles = [f for f in os.listdir(raw_data_path) if os.path.isfile(os.path.join(raw_data_path, f))]

    # training labels are given in cvs file
    label_csv_path = os.path.join(data_base_dir, 'fairface_label_' + data_split_subset + '.csv')
    label_df = pd.read_csv(label_csv_path)
    lb_make = LabelEncoder()
    # transform string to int
    label_df[protected_variable+"_int"] = lb_make.fit_transform(label_df[protected_variable])
    train_labels = label_df[protected_variable+"_int"].values

    class_idx_and_names = np.unique(label_df[protected_variable+"_int"].astype(str) + ";" + label_df[protected_variable].astype(str))
    class_idx_to_name = np.array([s.split(';') for s in class_idx_and_names])
    # class_idx_to_name = dict(zip(class_idx_to_name[:,0],class_idx_to_name[:,1]))

    corpus_file_content = []
    batch_id = 0
    for c_id, c_name in class_idx_to_name:
        class_example_ids = np.argwhere(train_labels == int(c_id))[:,0]

        random_class_ids = class_example_ids[np.random.choice(len(class_example_ids),
                                                              n_random_per_class)]

        batch_start_idx = 0

        while batch_start_idx < n_random_per_class:
            # load (up to) batch_size examples

            class_example_ids_batch = class_example_ids[batch_start_idx:batch_start_idx+batch_size]
            image_batch = list()
            for class_example_id in class_example_ids_batch:
                image_array = image_path_to_numpy(os.path.join(data_base_dir,
                                                               label_df["file"][class_example_id]))
                image_batch.append(image_array)
            image_batch = np.stack(image_batch)

            batch_name = 'batch' + str(batch_id).zfill(4)
            np.save(os.path.join(data_out_dir, batch_name), image_batch)
            corpus_file_content.append([batch_name + '.npy', c_name, str(c_id)])

            batch_start_idx = batch_start_idx + batch_size
            batch_id = batch_id + 1

    corpus_file_content = np.array(corpus_file_content)
    np.savetxt(os.path.join(data_out_dir, 'corpus.csv'), corpus_file_content, delimiter=",", fmt='%s')
