import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, AveragePooling2D, Lambda, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Split the input into three groups along the channel dimension
    inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)

    # Apply 1x1 convolution to each group
    conv1_group1 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
    conv1_group2 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
    conv1_group3 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])

    # Downsampling using Average Pooling
    pooled_group1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_group1)
    pooled_group2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_group2)
    pooled_group3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_group3)

    # Concatenate the pooled groups along the channel dimension
    concatenated = Concatenate()([pooled_group1, pooled_group2, pooled_group3])

    # Flatten the concatenated feature maps
    flatten_layer = Flatten()(concatenated)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model