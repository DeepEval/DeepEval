import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels

    # Branch 1: 1x1 Convolution for Dimensionality Reduction
    branch1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Branch 2: 1x1 and 3x3 Convolution for Feature Extraction
    branch2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

    # Branch 3: 1x1 and 5x5 Convolution for Feature Extraction
    branch3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch3)

    # Branch 4: 3x3 Max Pooling followed by 1x1 Convolution
    branch4 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(input_layer)
    branch4 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4)

    # Concatenate outputs of branches
    concat = Concatenate()([branch1, branch2, branch3, branch4])

    # Batch Normalization and Flatten
    bn = BatchNormalization()(concat)
    flatten = Flatten()(bn)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model

# Example usage:
model = dl_model()