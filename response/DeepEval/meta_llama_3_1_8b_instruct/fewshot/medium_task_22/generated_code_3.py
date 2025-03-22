import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 3x3 convolutions
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)

    # Branch 2: 1x1 convolutions followed by two 3x3 convolutions
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    # Branch 3: Max pooling
    max_pool3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_layer)

    # Concatenate the outputs from all branches
    branch_output = Concatenate()([max_pool1, max_pool2, max_pool3])

    # Flatten the concatenated feature maps
    flatten = Flatten()(branch_output)

    # Apply two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model