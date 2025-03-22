import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    # Define input layer with shape (32, 32, 3) for CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))

    # Define convolutional path 1 (1x1 convolution)
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Define convolutional path 2 (2 3x3 convolutions)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)

    # Define convolutional path 3 (1x1 convolution + 3x3 convolution)
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)

    # Define convolutional path 4 (max pooling + 1x1 convolution)
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    conv4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(maxpool)

    # Concatenate outputs from all paths
    outputs = Concatenate()([conv1, conv2, conv3, conv4])

    # Flatten and apply batch normalization
    flat = Flatten()(outputs)
    batch_norm = BatchNormalization()(flat)

    # Apply dense layers with 128 units and softmax activation for classification
    dense1 = Dense(units=128, activation='relu')(batch_norm)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Define model and return
    model = keras.Model(inputs=input_layer, outputs=dense2)
    return model