import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, AveragePooling2D, Flatten, Dense, concatenate
from tensorflow.keras.models import Model

def rgb_split(x):
    # Split the input image into three groups along the channel dimension
    x_r, x_g, x_b = tf.split(x, num_or_size_splits=3, axis=3)
    return x_r, x_g, x_b

def rgb_merge(rgb_outputs):
    # Concatenate the three groups of feature maps along the channel dimension
    return tf.concat(rgb_outputs, axis=3)

def dl_model(num_classes):
    # Input layer
    img_input = Input(shape=(32, 32, 3))

    # Split the input image into three groups
    x_r, x_g, x_b = Lambda(rgb_split)(img_input)

    # Apply 1x1 convolutions to each group
    conv_r = Conv2D(32 // 3, (1, 1), padding='same')(x_r)
    conv_g = Conv2D(32 // 3, (1, 1), padding='same')(x_g)
    conv_b = Conv2D(32 // 3, (1, 1), padding='same')(x_b)

    # Downsample each group
    pool_r = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_r)
    pool_g = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_g)
    pool_b = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_b)

    # Concatenate the three groups of feature maps
    concat = Lambda(rgb_merge)([pool_r, pool_g, pool_b])

    # Flatten the concatenated feature maps
    flattened = Flatten()(concat)

    # Fully connected layers
    dense1 = Dense(256, activation='relu')(flattened)
    dense2 = Dense(num_classes, activation='softmax')(dense1)

    # Model creation
    model = Model(img_input, dense2)

    return model

# Example usage:
num_classes = 10  # Number of classes in CIFAR-10 dataset
model = build_model(num_classes)