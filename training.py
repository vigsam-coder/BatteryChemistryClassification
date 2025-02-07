import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from transformers import get_linear_schedule_with_warmup
from model import build_model
from data_loader import load_and_split_data, regular_encode, create_tf_datasets,load_tokenizer
from utils import load_config

# Load configuration
config = load_config()

# Optimizer and scheduler setup
def get_optimizer():
    return Adam(learning_rate=config['learning_rate'])


def get_scheduler(optimizer, steps_per_epoch, warmup_steps):
    total_steps = steps_per_epoch * config['epochs']
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

def get_decay():
    lr_schedule = ExponentialDecay(
        initial_learning_rate=config['learning_rate'],
        decay_steps=config["warmup_steps"],
        decay_rate=config["weight_decay"],
        staircase=True)
    return lr_schedule


# Callback functions
def get_callbacks():
    checkpoint_dir = config['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'best_model.h5'),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config['early_stopping_patience'],
        restore_best_weights=True
    )

    lr = LearningRateScheduler(get_decay())

    return [checkpoint_callback, early_stopping,lr]


def train():
    # Load dataset
    train_df, val_df = load_and_split_data(config['data_path'])
    tokenizer = load_tokenizer(config['model_name'])

    # Encoding data
    x_train = regular_encode(train_df['text'].tolist(), tokenizer, maxlen=config['max_len'])
    x_val = regular_encode(val_df['text'].tolist(), tokenizer, maxlen=config['max_len'])

    y_train = train_df.iloc[:, :-1].values
    y_val = val_df.iloc[:, :-1].values

    # Create TensorFlow datasets
    train_dataset, val_dataset = create_tf_datasets(x_train, y_train, x_val, y_val, config['batch_size'])

    # Build and compile model
    model = build_model(max_len=config['max_len'])
    optimizer = get_optimizer()
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    # Train the model
    model.fit(
        train_dataset,
        steps_per_epoch=train_df.shape[0] // config['batch_size'],
        validation_data=val_dataset,
        validation_steps=val_df.shape[0] // config['batch_size'],
        epochs=config['epochs'],
        callbacks=get_callbacks()
    )


if __name__ == '__main__':
    train()
