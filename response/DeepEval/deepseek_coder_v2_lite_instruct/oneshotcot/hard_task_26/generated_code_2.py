import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add, Lambda
from keras.models import Model
import tensorflow as tf

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch paths
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    branch2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(main_path)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Lambda(lambda x: tf.image.resize(x, (32, 32)))(branch2)
    branch3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(main_path)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Lambda(lambda x: tf.image.resize(x, (32, 32)))(branch3)

    # Concatenate outputs from branch paths
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Pass through a 1x1 convolutional layer
    main_path_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)

    # Add branch path to main path
    added = Add()([main_path_output, main_path])

    # Batch normalization and flatten
    batch_norm = BatchNormalization()(added)
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model