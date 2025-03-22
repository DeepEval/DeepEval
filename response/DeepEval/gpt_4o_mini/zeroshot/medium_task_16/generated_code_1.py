import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, AveragePooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Input layer for CIFAR-10 images (32x32x3)
    input_layer = Input(shape=(32, 32, 3))

    # Split the input along the channel dimension into 3 groups
    split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Create a list to hold the processed feature maps
    processed_maps = []

    # Process each group with a 1x1 convolution and average pooling
    for group in split_channels:
        # Apply 1x1 convolution with one-third of the input channels
        conv_layer = Conv2D(filters=1, kernel_size=(1, 1), activation='relu')(group)
        # Apply average pooling to downsample
        pooled_layer = AveragePooling2D(pool_size=(2, 2))(conv_layer)
        processed_maps.append(pooled_layer)

    # Concatenate the processed feature maps along the channel dimension
    concatenated = Concatenate(axis=-1)(processed_maps)

    # Flatten the concatenated feature maps into a 1D vector
    flatten_layer = Flatten()(concatenated)

    # Fully connected layers for classification
    dense_layer1 = Dense(128, activation='relu')(flatten_layer)
    dense_layer2 = Dense(10, activation='softmax')(dense_layer1)  # 10 classes for CIFAR-10

    # Create the model
    model = Model(inputs=input_layer, outputs=dense_layer2)

    return model

# Example of creating the model
model = dl_model()
model.summary()