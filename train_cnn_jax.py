
import jax.random as random
import numpy as np
import os
import tensorflow as tf
import jax
import optax
import jax.numpy as jnp
import flax.linen as nn

from flax.training import train_state
from typing import Any
from datetime import datetime
from jax import jit

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

os.environ.pop('TF_USE_LEGACY_KERAS', None)


class TrainState(train_state.TrainState):
    batch_stats: Any


BATCH_SIZE = 32
TRAIN_PERCENT = 0.8
VAL_PERCENT = 1.0 - TRAIN_PERCENT
LR = 1e-3
EPOCH_NUM = 10
IMAGE_SIZE = (224, 224)
train_dir = 'data_img/imagenette2/train'
val_dir = 'data_img/imagenette2/val'
# Define some constants
BATCH_SIZE = 256
IMG_SIZE = (224, 224)

# Load the data
train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset='both',
    image_size=(320, 320),
    batch_size=None,
    seed=123)


# Optional: Prepare dataset for performance
train_ds = train_ds.shuffle(1000)


def preprocess(image, label):
    image = tf.image.resize(image, (224, 224))
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

    # Ensure the image is of type float32
    image = tf.cast(image, tf.float32) / 255.0

    # Normalize the image
    normalized_image = (image - mean) / std
    return normalized_image, label


# # Prefetch to improve performance
train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)


class SimpleCNN(nn.Module):
    @nn.compact
    def __call__(self, x, train):
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))  # Flatten the feature maps
        x = nn.Dense(features=256)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = nn.Dense(features=37)(x)  # Output layer with 37 classes
        return x


# Instantiate the model
model = SimpleCNN()


def create_train_state(rng, model, total_steps):
    variables = model.init(rng, jnp.ones(
        [BATCH_SIZE, 224, 224, 3]), train=False)
    tx = optax.adam(learning_rate=LR)  # Initialize SGD Optimizer
    return TrainState.create(apply_fn=model.apply, params=variables['params'], batch_stats=variables['batch_stats'], tx=tx)


rng = random.PRNGKey(0)
total_steps = EPOCH_NUM*len(train_ds) + EPOCH_NUM
state = create_train_state(rng, model, total_steps)


@jit
def train_step(state, batch):
    def loss_fn(params):
        logits, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats}, x=batch[0], train=True, mutable=['batch_stats'])
        one_hot = jax.nn.one_hot(batch[1], num_classes=37)
        loss = optax.softmax_cross_entropy(
            logits=logits, labels=one_hot).mean()
        return loss, (logits, updates)

    (loss, (logits, updates)), grads = jax.value_and_grad(
        loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])
    metrics = {
        'loss': -loss,
        'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch[1]),
    }
    return state, metrics


@jit
def eval_step(state, batch):
    logits = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats}, x=batch[0], train=False, mutable=False)
    metrics = {
        'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch[1]),
    }
    one_hot = jax.nn.one_hot(batch[1], num_classes=37)
    loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
    return loss, metrics


start_time = datetime.now()


def tfds_to_numpy(ds):
    for batch in ds:
        images, labels = batch
        # Convert each batch to numpy arrays
        yield jax.device_put(np.array(images)), jax.device_put(np.array(labels))


for epoch in range(EPOCH_NUM):
    train_acc = 0
    train_loss = 0
    for batch in tfds_to_numpy(train_ds):
        state, metrics = train_step(state, batch)
        train_acc += metrics['accuracy']
        train_loss += metrics['loss']
    train_loss /= len(train_ds)
    train_acc /= len(train_ds)
    train_acc *= 100

    val_acc = 0
    val_loss = 0
    for batch in tfds_to_numpy(val_ds):
        tmp_loss = 0
        tmp_loss, metrics = eval_step(state, batch)
        val_loss += tmp_loss
        val_acc += metrics['accuracy']
    val_loss /= len(val_ds)
    val_acc /= len(val_ds)
    val_acc *= 100

    print(f'EPOCH {epoch + 1}: train loss: {train_loss} train acc:{
          train_acc} val_loss: {val_loss} val acc: {val_acc}')

end_time = datetime.now()
print("Whole training took:", (end_time - start_time).seconds, "s")
