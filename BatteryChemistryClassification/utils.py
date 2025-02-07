import json
import yaml
import numpy as np
import tensorflow as tf
import os

# Load configuration
def load_config(config_file="config/config.yaml"):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

# Set random seed
def seed_everything(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed_everything()
