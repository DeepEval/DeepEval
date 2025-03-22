import keras
from keras.layers import Input, Conv2D, Concatenate, Add, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 64))

    # Main path
    dim_reduction = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    feature_extraction1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(dim_reduction)
    feature_extraction2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(dim_reduction)
    concatenated = Concatenate()([feature_extraction1, feature_extraction2])

    # Branch path
    feature_extraction3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Combine the main and branch paths
    combined = Add()([concatenated, feature_extraction3])

    # Apply batch normalization
    batch_norm = BatchNormalization()(combined)

    # Flatten the output
    flatten_layer = Flatten()(batch_norm)

    # Define the fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model