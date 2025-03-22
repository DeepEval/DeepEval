import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    conv_init = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch for local feature extraction
    conv1_local = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_init)
    conv2_local = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_local)

    # Branch with average pooling and upsampling
    conv1_global = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_init)
    conv2_global = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_global)
    conv3_global = Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv2_global)

    # Branch with average pooling, upsampling, and 1x1 convolution
    conv1_final = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(conv_init)
    conv2_final = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_final)
    conv3_final = Conv2DTranspose(filters=16, kernel_size=(7, 7), strides=(4, 4), padding='same', activation='relu')(conv2_final)
    conv4_final = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv3_final)

    # Concatenate outputs from branches
    concat = Concatenate()([conv1_local, conv2_local, conv2_global, conv3_global, conv4_final])

    # Fully connected layer for classification
    flatten = Flatten()(concat)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model