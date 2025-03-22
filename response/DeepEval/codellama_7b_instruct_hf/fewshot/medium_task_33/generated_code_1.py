import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Split, SeparableConv2D, Flatten, Dense, Softmax
from tensorflow.keras.models import Model

def cifar_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input image into three channel groups
    channel_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)

    # Process each channel group through a separable convolutional layer of varying sizes
    conv1_1 = SeparableConv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(channel_groups[0])
    conv1_2 = SeparableConv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(channel_groups[1])
    conv1_3 = SeparableConv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(channel_groups[2])

    # Concatenate the outputs from the three groups
    concatenated = Lambda(lambda x: tf.concat(values=x, axis=-1))([conv1_1, conv1_2, conv1_3])

    # Flatten the output and pass through three fully connected layers
    flattened = Flatten()(concatenated)
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model