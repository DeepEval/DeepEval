import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, AveragePooling2D, Add, GlobalAveragePooling2D, Dense, Reshape, Flatten
from keras.models import Model

def dl_model():
    # Input layer for CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))

    # First block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    avg_pool = AveragePooling2D(pool_size=(2, 2))(conv2)

    # Combine input layer with the output of the first block
    added_layer = Add()([input_layer, avg_pool])

    # Second block
    global_avg_pool = GlobalAveragePooling2D()(added_layer)
    dense1 = Dense(units=32, activation='relu')(global_avg_pool)
    dense2 = Dense(units=32, activation='sigmoid')(dense1)  # Channel weights

    # Reshape the weights to match the input channels
    reshaped_weights = Reshape((1, 1, 32))(dense2)

    # Multiply the input with the refined weights
    multiplied_output = tf.multiply(added_layer, reshaped_weights)

    # Flatten the output and pass through a final fully connected layer
    flatten_output = Flatten()(multiplied_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_output)

    # Constructing the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model