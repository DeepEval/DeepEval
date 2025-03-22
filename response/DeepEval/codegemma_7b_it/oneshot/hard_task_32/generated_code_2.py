import keras
from keras import layers
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, BatchNormalization, Flatten, Dense, Dropout

def specialized_block(input_tensor):
    # Depthwise separable convolution
    depthwise = layers.SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)

    # 1x1 convolutional layer to extract features
    pointwise = layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise)

    return pointwise

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Branch 1
    branch1 = specialized_block(input_tensor=input_layer)
    branch1 = layers.Dropout(rate=0.2)(branch1)

    # Branch 2
    branch2 = specialized_block(input_tensor=input_layer)
    branch2 = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch2)
    branch2 = layers.Dropout(rate=0.2)(branch2)

    # Branch 3
    branch3 = layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch3)
    branch3 = layers.Dropout(rate=0.2)(branch3)

    # Concatenate outputs from branches
    concatenated = concatenate([branch1, branch2, branch3])

    # Batch normalization
    bn = layers.BatchNormalization()(concatenated)

    # Flatten
    flatten_layer = Flatten()(bn)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model