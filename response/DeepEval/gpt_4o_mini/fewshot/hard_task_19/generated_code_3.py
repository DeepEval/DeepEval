import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Add, Multiply, Reshape, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv2)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

    # Branch path
    gap = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=128, activation='relu')(gap)
    dense2 = Dense(units=32, activation='sigmoid')(dense1)  # Channel weights
    reshaped_weights = Reshape((1, 1, 32))(dense2)  # Reshape to match input dimensions

    # Multiply branch output with input
    weighted_input = Multiply()([input_layer, reshaped_weights])

    # Combine both paths
    combined_output = Add()([main_path, weighted_input])

    # Final classification layers
    flatten = Flatten()(combined_output)
    final_dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(final_dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model