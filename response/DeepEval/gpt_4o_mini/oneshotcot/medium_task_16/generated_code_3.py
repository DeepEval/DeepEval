import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Split the input into three groups along the channel dimension
    split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Apply 1x1 convolutions to each group
    conv_layers = []
    for tensor in split_tensors:
        # Determine the number of filters for each group (one-third of input channels)
        num_filters = tensor.shape[-1] // 3
        conv = Conv2D(filters=num_filters, kernel_size=(1, 1), activation='relu')(tensor)
        conv_layers.append(conv)

    # Downsample each group using average pooling
    pooled_layers = []
    for conv in conv_layers:
        pooled = AveragePooling2D(pool_size=(2, 2))(conv)
        pooled_layers.append(pooled)

    # Concatenate the pooled outputs along the channel dimension
    concatenated = Concatenate(axis=-1)(pooled_layers)

    # Flatten the concatenated feature maps
    flatten_layer = Flatten()(concatenated)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model