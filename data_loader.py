import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, RobertaTokenizer, BertTokenizer, LongformerTokenizer
from utils import load_config

# Load configuration
config = load_config()


# Dataset preparation
def load_and_split_data(file_path):
    df = pd.read_csv(file_path)
    df = df.iloc[:, 1:]  # Remove the first column
    np.random.seed(42)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    split_idx = int(len(df) * 0.8)  # 80% train, 20% val
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    return train_df, val_df


# Tokenizer loading
def load_tokenizer(model_name):
    if model_name == "roberta-base":
        return RobertaTokenizer.from_pretrained("roberta-base")
    elif model_name == "bert-base-uncased":
        return BertTokenizer.from_pretrained("bert-base-uncased")
    elif model_name == "allenai/longformer-base-4096":
        return LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    elif model_name == "microsoft/deberta-large":
        return AutoTokenizer.from_pretrained("microsoft/deberta-large")
    else:
        raise ValueError("Unsupported model name")


# Encoding function
def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts,
        max_length=maxlen,
        padding='max_length',
        truncation=True,
        return_tensors="tf"
    )

    token_type_ids = enc_di.get('token_type_ids', None)
    if token_type_ids is not None:
        return np.array(enc_di['input_ids'], dtype=np.int32), np.array(enc_di['attention_mask'],
                                                                       dtype=np.int32), np.array(token_type_ids,
                                                                                                 dtype=np.int32)
    else:
        return np.array(enc_di['input_ids'], dtype=np.int32), np.array(enc_di['attention_mask'], dtype=np.int32)


def create_tf_datasets(x_train, y_train, x_val, y_val, batch_size):
    AUTO = tf.data.experimental.AUTOTUNE
    train_dataset = (
        tf.data.Dataset
        .from_tensor_slices((x_train, y_train))
        .shuffle(len(x_train))
        .batch(batch_size)
        .repeat()
        .prefetch(AUTO)
    )

    val_dataset = (
        tf.data.Dataset
        .from_tensor_slices((x_val, y_val))
        .shuffle(len(x_val))
        .batch(batch_size)
        .repeat()
        .prefetch(AUTO)
    )

    return train_dataset, val_dataset
