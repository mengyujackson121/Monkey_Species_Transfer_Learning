
# From https://www.tensorflow.org/tutorials/load_data/images
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def load_data_sets(data_dir, img_height, img_width, batch_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir + "/train",
        label_mode="categorical",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir + "/val",
            label_mode="categorical",

      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)

def load_model(model_name, num_classes):
    if model_name == "xception":
        base_model = keras.applications.Xception(
            weights="imagenet",  # Load weights pre-trained on ImageNet.
            input_shape=(150, 150, 3),
            include_top=False,
        )  # Do not include the ImageNet classifier at the top.

        # Freeze the base_model
        base_model.trainable = False

        # Create new model on top
        inputs = keras.Input(shape=(150, 150, 3))
        # x = data_augmentation(inputs)  # Apply random data augmentation

        # Pre-trained Xception weights requires that input be normalized
        # from (0, 255) to a range (-1., +1.), the normalization layer
        # does the following, outputs = (inputs - mean) / sqrt(var)
        norm_layer = keras.layers.experimental.preprocessing.Normalization()
        mean = np.array([127.5] * 3)
        var = mean ** 2
        # Scale inputs to [-1, +1]
        #x = norm_layer(x)
        x = norm_layer(inputs)
        norm_layer.set_weights([mean, var])

        # The base model contains batchnorm layers. We want to keep them in inference mode
        # when we unfreeze the base model for fine-tuning, so we make sure that the
        # base_model is running in inference mode here.
        x = base_model(x, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
        outputs = keras.layers.Dense(num_classes)(x)
        model = keras.Model(inputs, outputs)
        return model
    else:
        raise ValueError(f"Unsupported model name {model_name}")

