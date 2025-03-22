import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dropout, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution + 3x3 convolution
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)

    # Branch 2: 1x1 convolution + 1x7 convolution + 7x1 convolution + 3x3 convolution
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

    # Branch 3: Max pooling
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)

    # Concatenate outputs from all branches
    concat = Concatenate()([branch1, branch2, branch3])

    # Batch normalization
    batch_norm = BatchNormalization()(concat)

    # Dropout
    dropout = Dropout(rate=0.25)(batch_norm)

    # Flatten
    flatten = Flatten()(dropout)

    # Dense layer 1
    dense1 = Dense(units=128, activation='relu')(flatten)

    # Dropout
    dropout2 = Dropout(rate=0.5)(dense1)

    # Dense layer 2
    dense2 = Dense(units=64, activation='relu')(dropout2)

    # Dense layer 3 (output)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])