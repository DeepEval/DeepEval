import keras
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, ReLU, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Convolutional layer
    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Batch normalization and ReLU activation
    conv_bn = BatchNormalization()(conv)
    conv_relu = ReLU()(conv_bn)

    # Global average pooling
    gavgp = MaxPooling2D(pool_size=(8, 8), strides=(8, 8), padding='same')(conv_relu)

    # Two fully connected layers
    fc1 = Dense(units=128, activation='relu')(gavgp)
    fc2 = Dense(units=10, activation='softmax')(fc1)

    # Reshape the output to match the size of the initial features
    fc2_reshaped = Flatten()(fc2)

    # Weighted feature maps
    weighted_features = keras.layers.multiply([input_layer, fc2_reshaped])

    # Concatenate the input layer and the weighted feature maps
    concatenated = keras.layers.concatenate([weighted_features, input_layer])

    # 1x1 convolution and average pooling
    conv_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)
    avgp = MaxPooling2D(pool_size=(8, 8), strides=(8, 8), padding='same')(conv_1x1)

    # Final fully connected layer
    fc3 = Dense(units=10, activation='softmax')(avgp)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=fc3)

    return model