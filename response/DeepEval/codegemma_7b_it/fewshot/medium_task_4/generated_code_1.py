import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1: Two blocks of convolution followed by average pooling
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1_1)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1_2)
    path1_output = Add()([pool1, pool2])

    # Path 2: Single convolutional layer
    conv2_1 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2_output = Flatten()(conv2_1)

    # Combine outputs from both pathways
    concat_output = keras.layers.concatenate([path1_output, path2_output])

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(concat_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model