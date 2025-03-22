import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Lambda, BatchNormalization, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(input_layer)

    # Split the input into three groups
    split1, split2, split3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)

    # Concatenate the outputs from the three groups
    concatenated = Concatenate(axis=3)([conv1x1, conv3x3, conv5x5])

    # Branch path
    branch1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)

    # Add the outputs of the main and branch paths
    fused = tf.add(concatenated, branch1x1)

    # Flatten the fused features
    flatten_layer = Flatten()(fused)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model