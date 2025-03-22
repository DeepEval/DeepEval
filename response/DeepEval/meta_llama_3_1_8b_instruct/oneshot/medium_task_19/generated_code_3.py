import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 dataset: 32x32x3
    conv = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)  # Branch 1: 1x1 convolution for dimensionality reduction
    branch2_conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)  # Branch 2: 1x1 convolution
    branch2_conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_conv1)  # Branch 2: 3x3 convolution
    branch3_conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)  # Branch 3: 1x1 convolution
    branch3_conv2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch3_conv1)  # Branch 3: 5x5 convolution
    branch4_maxpool = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(input_layer)  # Branch 4: 3x3 max pooling
    branch4_conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4_maxpool)  # Branch 4: 1x1 convolution
    output_tensor = Concatenate()([conv, branch2_conv2, branch3_conv2, branch4_conv1])  # Concatenate the outputs of four branches

    bath_norm = BatchNormalization()(output_tensor)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)  # Output layer with 10 units for 10-class classification

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model