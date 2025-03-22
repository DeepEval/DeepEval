import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First sequential block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    # Second sequential block
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling1)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    # Combine outputs from both sequential blocks
    merged = Concatenate()([conv1, max_pooling1, conv2, max_pooling2])

    # Add batch normalization and flatten the output
    batch_norm = BatchNormalization()(merged)
    flatten = Flatten()(batch_norm)

    # Create a fully connected layer with 128 units and ReLU activation
    dense1 = Dense(units=128, activation='relu')(flatten)

    # Create a fully connected layer with 64 units and ReLU activation
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Create a fully connected layer with 10 units and softmax activation
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model by passing the input layer and output layer
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model