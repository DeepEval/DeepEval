import keras
import tensorflow as tf
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three channel groups
    input_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)

    # Feature extraction for each group using separable convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_groups[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_groups[1])
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_groups[2])

    # Concatenate the outputs from each group
    concat_layer = Concatenate()([conv1, conv2, conv3])

    # Max pooling to reduce dimensionality
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(concat_layer)

    # Flatten the output for fully connected layers
    flatten_layer = Flatten()(max_pooling)

    # Fully connected layers for classification
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    dense2 = Dense(units=32, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model