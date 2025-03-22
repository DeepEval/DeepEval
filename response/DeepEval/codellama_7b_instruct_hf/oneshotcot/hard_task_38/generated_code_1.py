import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # First pathway
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    batch_norm1 = BatchNormalization()(max_pooling1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv4)

    # Second pathway
    conv5 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    max_pooling5 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv5)
    batch_norm5 = BatchNormalization()(max_pooling5)
    conv6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm5)
    conv7 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv6)
    conv8 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv7)
    max_pooling8 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv8)

    # Concatenate outputs from both pathways
    concatenated_outputs = Concatenate()([max_pooling2, max_pooling8])

    # Flatten and pass through two fully connected layers
    flattened_outputs = Flatten()(concatenated_outputs)
    dense1 = Dense(units=128, activation='relu')(flattened_outputs)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model