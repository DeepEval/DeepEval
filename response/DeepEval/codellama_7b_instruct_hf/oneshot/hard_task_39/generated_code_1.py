import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the first block
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
    max_pooling2 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(conv)
    max_pooling3 = MaxPooling2D(pool_size=(6, 6), strides=6, padding='valid')(conv)
    flat1 = Flatten()(max_pooling1)
    flat2 = Flatten()(max_pooling2)
    flat3 = Flatten()(max_pooling3)
    concatenate_layer = Concatenate()([flat1, flat2, flat3])

    # Define the second block
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenate_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concatenate_layer)
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(concatenate_layer)
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(concatenate_layer)
    output_layer = Concatenate()([conv1, conv2, conv3, maxpool])

    # Define the fully connected layers
    batch_norm = BatchNormalization()(output_layer)
    dense1 = Dense(units=128, activation='relu')(batch_norm)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model