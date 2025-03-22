import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Lambda, Flatten, Dense, Concatenate
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Apply 1x1 convolutions to each group (each group will have one-third of the input channels)
    conv_groups = [Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group) for group in split_layer]

    # Downsample each group using average pooling
    pooled_groups = [AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_group) for conv_group in conv_groups]

    # Concatenate the resulting groups along the channel dimension
    concatenated_features = Concatenate(axis=-1)(pooled_groups)

    # Flatten the concatenated feature maps into a one-dimensional vector
    flattened_features = Flatten()(concatenated_features)

    # Pass the flattened features through two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flattened_features)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model