import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate, GlobalMaxPooling1D, GlobalAveragePooling1D
from transformers import TFAutoModel
from utils import load_config

# Load configuration
config = load_config()


# Model building
def build_model(max_len=512):
    MODEL_NAME = config['model_name']

    if MODEL_NAME == "bert-base-uncased":
        input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
        input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
        segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")
        model = TFAutoModel.from_pretrained(MODEL_NAME)
        sequence_output = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids)[0]
    elif MODEL_NAME == 'microsoft/deberta-large':
        input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
        input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
        segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")
        model = TFAutoModel.from_pretrained(MODEL_NAME)
        sequence_output = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids)[0]
    else:
        input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
        input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
        model = TFAutoModel.from_pretrained(MODEL_NAME)
        sequence_output = model(input_ids, attention_mask=input_mask)[0]

    gp = GlobalMaxPooling1D()(sequence_output)
    ap = GlobalAveragePooling1D()(sequence_output)
    stack = concatenate([gp, ap], axis=1)
    out = Dense(config['num_classes'], activation='sigmoid')(stack)

    if MODEL_NAME in ["bert-base-uncased", 'microsoft/deberta-large']:
        model = Model(inputs=[input_ids, input_mask, segment_ids], outputs=out)
    else:
        model = Model(inputs=[input_ids, input_mask], outputs=out)

    return model
