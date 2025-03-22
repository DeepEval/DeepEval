import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Split the input into three groups along the channel dimension
    splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Apply 1x1 convolutions to each group
    conv1 = Conv2D(filters=1, kernel_size=(1, 1), activation='relu')(splits[0])
    conv2 = Conv2D(filters=1, kernel_size=(1, 1), activation='relu')(splits[1])
    conv3 = Conv2D(filters=1, kernel_size=(1, 1), activation='relu')(splits[2])

    # Downsample each group with average pooling
    pooled1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    pooled2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    pooled3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)

    # Concatenate the pooled feature maps along the channel dimension
    concatenated = Concatenate(axis=-1)([pooled1, pooled2, pooled3])

    # Flatten the concatenated feature maps
    flatten_layer = Flatten()(concatenated)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # 10 classes for CIFAR-10

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model